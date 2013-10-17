#define PI_F 3.141592653589793f

// Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

//CUDA utilities and system inlcudes
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "helper_cuda_gl.h"
#include <rendercheck_gl.h>
#include <math.h>
//Includes
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm>

#include <iostream>
#include <fstream>


#include "particles.h"
#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"
#include "particles_kernel.h"
#include "new_kern.h"
#include "utilities.h"


uint width = 600, height = 600;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -2.2e-3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3e-3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;

FILE* datalog;
FILE* crashlog;
const char* filename = "filenameunsetasd";
char logfile [256];
char crashname[231];


int mode = 0;
bool displayEnabled = true;
bool rotatable = false;
uint recordInterval = 0;
uint logInterval = 0;
uint partlogInt = 0;
bool bPause = false;
bool displaySliders = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 300;

SimParams pdata;

// simulation parameters
float timestep = 500; //in units of nanoseconds
double simtime = 0.0f;
float externalH = 100; //kA/m
float colorFmax = 0.75;
float clipPlane = -1.0;
float iter_dxpct = 0.035;
float rebuild_pct = 0.1;
float strain = 0, period = 24;//compression params, period in ms
float contact_dist = 1.05f;
float pin_dist = 0.995f;
float3 worldSize;
float maxtime = 0;
float force_dist = 8;

int resolved = 0;//number of times the integrator had to resolve a step

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *paramlist;

// Auto-Verification Code
int frameCount = 0;
bool g_useGL = true;


// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)


extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);

void cleanup()
{
	sdkDeleteTimer(&timer);    
}
void setParams(){
    psystem->setGlobalDamping(pdata.globalDamping);
    psystem->setRepelSpring(pdata.spring);
    psystem->setShear(pdata.shear);
    psystem->setExternalH(make_float3(0.0f, externalH*1e3, 0.0f));
	psystem->setColorFmax(colorFmax);
	psystem->setViscosity(pdata.viscosity);
	psystem->setDipIt(pdata.mutDipIter);
	psystem->setPinDist(pin_dist);
	psystem->setContactDist(contact_dist);
	psystem->setRebuildDist(rebuild_pct);
	psystem->setClipPlane(clipPlane);
}


void qupdate()
{
 	setParams();
	//this crude hack makes pinned particles at start unpinned so they can space and unfuck each other
	if(simtime < timestep)	psystem->setPinDist(0.8f);

	if(strain != 0) {
		float ws = worldSize.y*(1.0f + strain*sin(2.0f*PI_F/(period*1e6f)*simtime));
		psystem->dangerousResize(ws);
	}

	float dtout = 1e9*psystem->update(timestep*1e-9f, iter_dxpct);
	if(fabs(dtout - timestep) > .01f*dtout)
		resolved++;
	if(logInterval != 0 && frameCount % logInterval == 0){
		printf("iter %d at %.2f/%.1f us\n", frameCount, simtime*1e-3f, maxtime);
		psystem->logStuff(datalog, simtime*1e-3);	
		fflush(datalog);
	}

	if(frameCount % 1000 == 0 && frameCount != 0 && logf != 0){
		crashlog = fopen(crashname, "w");
		fprintf(crashlog, "Time: %.12g ns\n", simtime);
		psystem->logParams(crashlog);
		psystem->logParticles(crashlog);
		fclose(crashlog);//sets up the overwrite
	}

	if( (partlogInt!=0) && (frameCount % partlogInt == 0)){
		char pname [256];
		sprintf(pname, "/home/steve/Datasets/%s_plog%.5d.dat", filename, frameCount/partlogInt);
		FILE* plog = fopen(pname, "w");
		fprintf(plog, "Time: %g ns\n", simtime);
		psystem->logParams(plog);
		psystem->logParticles(plog);
		fclose(plog);	
	}
	

	simtime += dtout;//so that it logs at the correct time
	frameCount++;
}


// initialize OpenGL
void initGL(int argc, char **argv)
{  
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA MR Fluid Sim");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0, 1.3, 1.3, 1.0);
	glMatrixMode(GL_MODELVIEW);
    glutReportErrors();
}

void runBenchmark()
{
    printf("Run %u particles simulation for %f us...\n\n", pdata.numBodies, maxtime);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);
	while(simtime < maxtime*1e3f){
		qupdate();	    
	}
	//do a final write to the logfile
	if(logInterval > 0)
		psystem->logStuff(datalog, simtime*1e-3f);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
	float fAvgSeconds = (float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)frameCount;

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5fs, Size = %u particles\n", 
    		(1.0e-3 * pdata.numBodies)/fAvgSeconds, fAvgSeconds*frameCount, pdata.numBodies);

	if(logf != 0) {
		crashlog = fopen(crashname, "w");
		fprintf(crashlog, "Time: %.12g ns\n", simtime);
		psystem->logParams(crashlog);
		psystem->logParticles(crashlog);
		fclose(crashlog); //sets up the overwrite
	}
}

void computeFPS()
{
    fpsCount++;
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA MR Fluid Sim (%d particles): %3.1f fps Time: %.2f us",pdata.numBodies, ifps, simtime*1e-3);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 

        sdkResetTimer(&timer);  
    }
}


void drawAxes() 
{
	glPushMatrix();
	glViewport(0,0,width/6,height/6);	
	
	float axes_size = ((float)width/600)*worldSize.x/2;
	glTranslatef(0, 0.1*axes_size, -20.0f*axes_size);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glPushMatrix();
	//if(!rotatable) //this makes for prettier pictures but ugly movies
	glTranslatef(-worldSize.x/2,-worldSize.y/2,0);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); 
    glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);

	glLineWidth(1.0);
	glBegin (GL_LINES);
	glColor3f (0.1,0.1,0.1); 
	glVertex3f (0,0,0); glVertex3f (axes_size,0,0); 
	glVertex3f (0,0,0); glVertex3f (0,axes_size,0);
	glVertex3f (0,0,0); glVertex3f (0,0,axes_size); 
	glEnd();

	glRasterPos3f(1.5*axes_size,0,0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
    glRasterPos3f(0,1.5*axes_size,0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
    glRasterPos3f(0,0,1.5*axes_size);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');
	
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);

	glPopMatrix();
	glPopMatrix();
}

void display()
{
  
  	sdkStartTimer(&timer);  
	setParams();
    // update the simulation
    if (!bPause)
    {
		qupdate();
		if (renderer){ 
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), 
					psystem->getNumParticles());
		}
    }

    // render
   	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

	for (int c = 0; c < 3; ++c) {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
	
    // view transform
    //glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
		
	drawAxes();
	
	//draw cube and particles
	glPushMatrix();	
    
	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glViewport(0,0,width,height); //draw over full streen
	//draw particles
	renderer->display(ParticleRenderer::PARTICLE_SPHERES);

    // cube w/ anti-aliasing
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);
	glColor3f(0.0, 0.0, 0.0);
    glLineWidth(2.0);
	glutWireCube(worldSize.x);

	if (strain != 0.0){
		float3 ws_act = psystem->getWorldSize();//actual worldSize
		glColor3f(1.0,0.0,0.0);
		glBegin(GL_LINE_LOOP);
		glVertex3f(ws_act.x/2, ws_act.y/2, ws_act.z/2);
		glVertex3f(ws_act.x/2, -ws_act.y/2, ws_act.z/2);
		glVertex3f(ws_act.x/2, -ws_act.y/2, -ws_act.z/2);	
		glVertex3f(ws_act.x/2, ws_act.y/2, -ws_act.z/2);
		glEnd();
		glBegin(GL_LINE_LOOP);
		glVertex3f(-ws_act.x/2, ws_act.y/2, ws_act.z/2);
		glVertex3f(-ws_act.x/2, -ws_act.y/2, ws_act.z/2);
		glVertex3f(-ws_act.x/2, -ws_act.y/2, -ws_act.z/2);	
		glVertex3f(-ws_act.x/2, ws_act.y/2, -ws_act.z/2);
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(-ws_act.x/2, ws_act.y/2, ws_act.z/2);
		glVertex3f(ws_act.x/2, ws_act.y/2, ws_act.z/2);
		glVertex3f(-ws_act.x/2, -ws_act.y/2, ws_act.z/2);
		glVertex3f(ws_act.x/2, -ws_act.y/2, ws_act.z/2);
		glVertex3f(-ws_act.x/2, ws_act.y/2, -ws_act.z/2);
		glVertex3f(ws_act.x/2, ws_act.y/2, -ws_act.z/2);
		glVertex3f(-ws_act.x/2, -ws_act.y/2, -ws_act.z/2);
		glVertex3f(ws_act.x/2, -ws_act.y/2, -ws_act.z/2);
		glEnd();
	}
	
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);

	
	
	glPopMatrix();	


	if(recordInterval != 0 && (frameCount % recordInterval == 0)){
		char asd [32];
		sprintf(asd, "./cap/frame%.5d.ppm", frameCount/recordInterval);
		g_CheckRender->readback(width,height);
		g_CheckRender->savePPM((const char*)asd,true,NULL);
	}
    if (displaySliders) {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        glLineWidth(1.0);
		paramlist->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);  
	
    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
	if((maxtime > 0) && (simtime >= maxtime*1e3)){
		exit(0);	
	}

}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}


void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspRatio = (float) w / (float) h;
	gluPerspective(20.0, aspRatio, 0.1e-6, 10);
	
	width = w;
	height = h;

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer) {
        renderer->setWindowSize(w, h);
        renderer->setFOV(20);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x; oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders) {
        if (paramlist->Mouse(x, y, button, state)) {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
  r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
  r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
  r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
  r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (displaySliders) {
        if (paramlist->Motion(x, y)) {
            ox = x; oy = y;
            glutPostRedisplay();
            return;
        }
    }

	if (buttonState == 3) {
		// left+middle = zoom
		camera_trans[2] += (dy / 100) * 0.5 * fabs(camera_trans[2]);
	} 
	else if (buttonState & 2) {
		// middle = translate
		camera_trans[0] += dx / 6e4f;
		camera_trans[1] -= dy / 6e4f;
	}
	else if (buttonState & 1) {
		// left = rotate
		camera_rot[0] += dy / 5.0;
		camera_rot[1] += dx / 5.0;
	}

    ox = x; oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{

	uint nedges=0, ngraphs=0, vedges=0, vgraphs=0, hedges=0, hgraphs=0;
	switch (key) 
    {
    case ' ':
        bPause = !bPause;
        break;
    case 13:
		qupdate();
		if (renderer)
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        break;
    case '\033':
    case 'q':
       	exit(0);
        break;
    case 'm':
        psystem->getMagnetization();
		break;
   	case 'i':
		printf("interactions: %d\n", psystem->getInteractions());
		break;
	case 'n':
		psystem->NListStats();
		break;
	case 'd':
        psystem->densDist();
        break;
	case 's': 
		psystem->printStress();
		break;
	case 'g':
		psystem->getGraphData(ngraphs,nedges,vedges,vgraphs,hedges,hgraphs);
		printf("ngraphs: %d, nchainl: %f\t Edges = %d Mean edges = %f\n", ngraphs,
				(float)pdata.numBodies/(float) ngraphs, nedges, (float)nedges/(float)pdata.numBodies);
		printf("vgraphs: %d, vchainl: %f\t VEdges = %d Mean vedges = %f\n", vgraphs,
				(float)pdata.numBodies/(float) vgraphs, vedges, (float)vedges/(float)pdata.numBodies);
		printf("hgraphs: %d, hchainl: %f\t HEdges = %d Mean hedges = %f\n", hgraphs,
				(float)pdata.numBodies/(float) hgraphs, hedges, (float)hedges/(float)pdata.numBodies);
		break;
    case 'u':
        psystem->dumpParticles(0, pdata.numBodies);
        break;
    case 'r':
        displayEnabled = !displayEnabled;
        break;
	case '1':
        psystem->resetParticles(1100, 0.3f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '2':
        psystem->resetParticles(300, 1.0f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '3':
        psystem->resetParticles(20, 1.0f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
	case '5':  									// preset angle for pretty screenshots
		camera_rot[0] = 15;						// elevation angle
		camera_rot[1] = 23;					// azimuth angle
		camera_trans[0] = -0.0*worldSize.x;	// horiz shift
		camera_trans[1] = 0.05*worldSize.y;		// vert shift
		camera_trans[2] = -4.8*worldSize.z;		// zoom
		break;
	case '6':
		camera_rot[0] = 5;
		camera_rot[1] = 87;
		camera_trans[0] = 0;
		camera_trans[1] = 0;
		camera_trans[2] = -4.5*worldSize.z;
		break;
	case '7':
		rotatable = !rotatable;
		printf("Idle rotation toggled: %s\n", rotatable ? "on" : "off");	
		break;
	case 'h':
        displaySliders = !displaySliders;
        break;
	case 'b':
		psystem->getBadP();
	}

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders) {
        paramlist->Special(k, x, y);
    }
    demoMode = false;
    idleCounter = 0;
}

void idle(void)
{
    //printf("idlecounter: %d\t\r", idleCounter);
	if ((idleCounter++ > idleDelay) && (demoMode==false)) {
        demoMode = true;
        printf("Entering demo mode\n");
    }

    if (demoMode && rotatable) {
        camera_rot[1] += 0.02f;
    }

    glutPostRedisplay();
	
}

void initParamList()
{
    if (g_useGL) {
        // create a new parameter list
		paramlist = new ParamListGL("misc");
        paramlist->AddParam(new Param<float>("time step (ns)", timestep, 0.0, 1200, 5, &timestep));
		paramlist->AddParam(new Param<float>("spring constant",pdata.spring, 0, 100, 1, &pdata.spring));
		paramlist->AddParam(new Param<float>("H (kA/m)", externalH, 0, 1e3, 5, &externalH));
		paramlist->AddParam(new Param<float>("shear rate", pdata.shear, 0, 2000, 50, &pdata.shear));
		paramlist->AddParam(new Param<float>("colorFmax", colorFmax, 0, 1, 0.01f, &colorFmax));
    	paramlist->AddParam(new Param<float>("visc", pdata.viscosity, 0.001f, .25f, 0.001f, &pdata.viscosity));
		paramlist->AddParam(new Param<float>("max dx pct", iter_dxpct, 0, .1f, 0.001f, &iter_dxpct));
		paramlist->AddParam(new Param<float>("pin dist", pin_dist, 0.995f, 1.5f, 0.005f, &pin_dist));
		paramlist->AddParam(new Param<float>("contact_dist", contact_dist, .95f, 1.25f, 0.001f, &contact_dist));
		paramlist->AddParam(new Param<float>("rebuild dist", rebuild_pct, 0.0f, 1.0f, 0.005f, &rebuild_pct));
		paramlist->AddParam(new Param<float>("clip plane", clipPlane, -1.0, 1.0, 0.01, &clipPlane));
	}
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{

	//my addition of an anon function for setting values from cmd line
	auto clArgInt = [=](const char* label, unsigned int& val) { 
		if(checkCmdLineFlag(argc, (const char **) argv, label))
			val = getCmdLineArgumentInt(argc, (const char**)argv, label);
	};

	auto clArgFloat = [=](const char* label, float& val) { 
		if(checkCmdLineFlag(argc, (const char **) argv, label))
			val = getCmdLineArgumentFloat(argc, (const char**)argv, label);
	};

	//set the device and gl status
	unsigned int devID = gpuGetMaxGflopsDeviceId();
	if(checkCmdLineFlag(argc, (const char **)argv, "noGL")) {
		g_useGL = false;
		clArgInt("device", devID);
	}
	cudaSetDevice(devID);
	printf("devID: %d\n", devID);
	
	//set logging parameters
	clArgInt("record", recordInterval);
	clArgInt("logf", logInterval);
	clArgInt("plogf", partlogInt);	
	char* title;//i have no idea how this works	
	if(getCmdLineArgumentString( argc, (const char**)argv, "logt", &title)){
		filename = title;
	}
	
   	sprintf(logfile, "/home/steve/Datasets/%s.dat", filename);
	sprintf(crashname, "/home/steve/Datasets/%s.crash.dat", filename);

	//set the random seed
	uint randseed = time(NULL);
	clArgInt("randseed", randseed);
	srand(randseed);
	printf("randseed: %d\n", randseed);
	
	
	//DEFINE SIMULATION PARAMETERS

	//set defaults
	pdata.shear = 500;
	pdata.spring = 50;
	externalH = 100;
	pdata.viscosity = 0.1;
	pdata.colorFmax = colorFmax;
	pdata.globalDamping = 0.8f; 
	pdata.cdamping = 0.03f;
	pdata.cspring = 10;
	pdata.boundaryDamping = -0.03f;
	pdata.mutDipIter = 0;

	//first check if we are restarting
	bool restart = checkCmdLineFlag(argc, (const char**)argv, "restart");
	if(restart){
		printf("Attempting to restart from %s\n", crashname);	
		
		crashlog = fopen(crashname, "r");
		if(crashlog == NULL) fprintf(stderr,"No such crashfile exists!\n\n");
		
		char buff[1024]; 
		int numlines = 0;
		int matches;
		float time;
		char verno[30];

		if(fgets(buff, 1024, crashlog) != NULL) {
			matches = sscanf(buff, "Time: %lg ns", &simtime);
			printf("matches = %d\t time: %g ns\n", matches, simtime);
		}

		if(fgets(buff, 1024, crashlog) != NULL){
			printf("%s", buff);
			matches = sscanf(buff, "Build Date: %*s %*d %*d %*d:%*d:%*d\t svn version: %s", verno);
			printf("matches = %d\t verno: %s\n", matches, verno);
			if(strncmp(VERSION_NUMBER, verno, 25))
				fprintf(stderr, "Warning, running data from version: %s on %s\n", verno, VERSION_NUMBER);
		}

		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "vfrtot: %*f\t v0: %f\t v1: %f\t v2: %f", &pdata.volfr[0], 
					&pdata.volfr[1], &pdata.volfr[2]);
			printf("matches = %d,\t v0: %.4f\t v1: %.4f\t v2: %.4f\n", matches, pdata.volfr[0], 
					pdata.volfr[1], pdata.volfr[2]);
		}

		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "ntotal: %d n0: %d n1: %d n2: %d",&pdata.numBodies, 
					&pdata.nump[0], &pdata.nump[1], &pdata.nump[2]);
			printf("matches = %d,\t ntotal: %d n0: %d n1: %d n2: %d\n",matches, pdata.numBodies, 
					pdata.nump[0], pdata.nump[1], pdata.nump[2]);
		}

		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, " mu0: %f mu1: %f mu2: %f", &pdata.mu_p[0], &pdata.mu_p[1], 
					&pdata.mu_p[2]);
			printf("matches = %d,\t mu0: %f mu1: %f mu2: %f\n", matches, pdata.mu_p[0], 
					pdata.mu_p[1], pdata.mu_p[2]);
		}

		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, " a0: %g a1: %g a2: %g", &pdata.pRadius[0], 
					&pdata.pRadius[1], &pdata.pRadius[2]);
			printf("matches = %d,\t a0: %g a1: %g a2: %g\n", matches, pdata.pRadius[0], 
					pdata.pRadius[1], pdata.pRadius[2]);
		}
			
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, " std0: %f std1: %f std2: %f", &pdata.rstd[0], 
					&pdata.rstd[1], &pdata.rstd[2]);
			printf("matches = %d,\t std0: %f std1: %f std2: %f\n", matches, pdata.rstd[0], 
					pdata.rstd[1], pdata.rstd[2]);
		}
		
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "grid: %d x%d x %d = %*d cells", &pdata.gridSize.x, 
					&pdata.gridSize.y, &pdata.gridSize.z);
			printf("matches = %d \t grid %d %d %d\n", matches, pdata.gridSize.x, pdata.gridSize.y, 
					pdata.gridSize.z);
		}
		
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "worldsize: %fmm x %fmm x %fmm", &worldSize.x, &worldSize.y, &worldSize.z);
			printf("matches = %d\t ws: %f %f %f\n", matches, worldSize.x, worldSize.y, worldSize.z);
		}
		worldSize = worldSize*1e-3;
			
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "spring: %f visc: %f Pin_d: %f contact_d: %f", &pdata.spring, 
					&pdata.viscosity, &pin_dist, &contact_dist); 
			printf("matches %d\t k %f visc %f pind %f contactd %f\n", matches, pdata.spring, 
					pdata.viscosity, pin_dist, contact_dist);
		}
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "rebuildDist: %f", &rebuild_pct);
			printf("matches %d rbd_d %f\n", matches, rebuild_pct);
		}
		if(fgets(buff, 1024, crashlog) != NULL){
			matches = sscanf(buff, "H.x: %*f H.y: %f H.z %*f", &externalH);
			externalH = externalH*1e-3f;
			printf("matches %d externalH %f\n", matches, externalH);
		}	   
		//keep the crashlog open to read the particles
	} else {
		float worldsize1d = .35;//units of mm
		clArgFloat("wsize", worldsize1d);
		worldSize = make_float3(worldsize1d*1e-3f, worldsize1d*1e-3f, worldsize1d*1e-3f);
		float volume = worldSize.x*worldSize.y*worldSize.z; 

		float radius = 4.0f;
		//haven't figured out a good way to do this in a loop
		clArgFloat("rad0", radius);
		pdata.pRadius[0] = radius*1e-6f; //median diameter
		pdata.volfr[0] = 0.30f;
		clArgFloat("vfr0", pdata.volfr[0]);
		pdata.mu_p[0] = 2000; //relative permeability
		clArgFloat("xi0", pdata.mu_p[0]);
		pdata.nump[0] = (pdata.volfr[0] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[0],3)); 
		pdata.rstd[0] = 0; //sigma0 in log normal distribution
		clArgFloat("std0", pdata.rstd[0]);	
		if(pdata.rstd[0] > 0){//eq 3.24 crowe
			pdata.nump[0] = (pdata.volfr[0]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[0],3)
						*exp(4.5f*pdata.rstd[0]*pdata.rstd[0]));
		}

		radius = 6.0f;
		clArgFloat("rad1", radius);
		pdata.pRadius[1] = radius*1e-6f;
		pdata.volfr[1] = 0.0f;
		clArgFloat("vfr1", pdata.volfr[1]);
		pdata.mu_p[1] = 2000;
		clArgFloat("xi1", pdata.mu_p[1]);
		pdata.nump[1] = (pdata.volfr[1] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[1],3)); 
		pdata.rstd[1] = 0;
		clArgFloat("std1", pdata.rstd[1]);
		if(pdata.rstd[1] > 0){//eq 3.24 crowe
			pdata.nump[1] = (pdata.volfr[1]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[1],3)
						*exp(4.5f*pdata.rstd[1]*pdata.rstd[1]));
		}
		
		radius = 25.0f;
		clArgFloat("rad2", radius);
		pdata.pRadius[2] = radius*1e-6f;
		pdata.volfr[2] = 0.0f;
		clArgFloat("vfr2", pdata.volfr[2]);
		pdata.mu_p[2] = 1;
		clArgFloat("xi2", pdata.mu_p[2]);
		pdata.nump[2] = (pdata.volfr[2] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[2],3)); 
		pdata.rstd[2] = 0;
		clArgFloat("std2", pdata.rstd[2]);
		if(pdata.rstd[2] > 0){//eq 3.24 crowe
			pdata.nump[2] = (pdata.volfr[2]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[2],3)
						*exp(4.5f*pdata.rstd[2]*pdata.rstd[2]));
		}
		
		pdata.numBodies = pdata.nump[0] + pdata.nump[1] + pdata.nump[2];
	}


	//now we take comman flags, overwriting default/restart values


	clArgFloat("shear", pdata.shear);
	clArgFloat("k", pdata.spring);

	clArgFloat("strain", strain);
	clArgFloat("Period", period);//input period in ms

	clArgFloat("H", externalH);
	pdata.externalH = make_float3(0,externalH*1e3f,0);

	clArgFloat("dt", timestep);//units of ns
	clArgFloat("maxtime", maxtime);//units of ns as well
	clArgFloat("visc", pdata.viscosity);

	clArgFloat("cspring", pdata.cspring);
	
	clArgFloat("pin_d", pin_dist);
	clArgFloat("contact_dist", contact_dist);
	clArgFloat("rebuild_dist", rebuild_pct);
	clArgFloat("iterdx",iter_dxpct); 
	clArgInt("dipit", pdata.mutDipIter);
		
	bool benchmark = checkCmdLineFlag(argc, (const char**) argv, "benchmark") != 0;

	pdata.worldOrigin = worldSize*-0.5f;

	clArgFloat("fdist", force_dist);	
	float cellSize_des = force_dist*pdata.pRadius[0];
 
	//gridSize.x = gridSize.y = gridSize.z = GRID_SIZE;
	pdata.gridSize.x = floor(worldSize.x/cellSize_des);
	pdata.gridSize.y = floor(worldSize.y/cellSize_des);
	pdata.gridSize.z = floor(worldSize.z/cellSize_des);

	pdata.cellSize.x = worldSize.x/pdata.gridSize.x;
	pdata.cellSize.y = worldSize.y/pdata.gridSize.y;
	pdata.cellSize.z = worldSize.z/pdata.gridSize.z;

	//INITIALIZE THE DATA STRUCTURES

    if (!g_useGL) {
        cudaInit(argc, argv);
    } else {
        initGL(argc, argv);
        cudaGLInit(argc, argv);
    }
	
	psystem = new ParticleSystem(pdata, g_useGL, worldSize);
	
	if (g_useGL) {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);

	initParamList();
	setParams();
	
	if(restart) {
		int val = psystem->loadParticles(crashlog);
		fclose(crashlog);
		if(val < 0) exit(0);
	} else {
		psystem->resetParticles(1100, 0.4f);
	}
    
	if (g_useGL) 
        initMenus();

	if( recordInterval != 0 ) {
        g_CheckRender = new CheckBackBuffer(width, height, 4);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }
	psystem->logParams(stdout); 
	
	if(logInterval != 0){
		printf("saving: %s\n",logfile);
		datalog = fopen(logfile, "a");
		if(datalog == NULL) {
			fprintf(stderr,"failed to open particle logfile, ferror %d\n", ferror(datalog));
		}
		if(!restart) {
			psystem->logParams(datalog);	
			fprintf(datalog, "time\tshear\textH\tchainl\tedges\ttopf\tbotf\tgstress\tkinen \tM.x \tM.y \tM.z \tvedge\tvgraph\thedge\thgraph\n");
		}
	}
	
	

    if (benchmark || !g_useGL) 
    {
       	if(maxtime <= 0)
			maxtime = timestep*500*1e-3;

        runBenchmark();
    }
	else if(g_useGL)
    {
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(key);
        glutSpecialFunc(special);
        glutIdleFunc(idle);

        atexit(cleanup);

        glutMainLoop();
    }
	printf("%d/%d iterations had to re-solved\n", resolved, frameCount);
	if(logInterval != 0)
		fclose(datalog);

    if (psystem)
        delete psystem;

    cudaThreadExit();

	exit(0);
    //shrEXIT(argc, (const char**)argv);
}
