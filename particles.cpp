#define PI_F 3.141592653589793f

// Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

//CUDA utilities and system inlcudes
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <rendercheck_gl.h>
#include <math.h>
#include <cutil_math.h>
//Includes
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm>
//#include <cuda_gl_interop.h>

#include "particles.h"
#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"
#include "particles_kernel.h"
#include "new_kern.h"

//shared library test functions
#include <shrUtils.h>
#include <shrQATest.h>

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#ifdef __DEVICE_EMULATION__
	const int binIdx = 1;	// pick the proper sReferenceBin
#endif

const uint width = 800, height = 600;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -.002};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

FILE* datalog;
FILE* crashlog;
const char* filename = "filenameunsetasd";
char logfile [256];
char crashname[231];


int mode = 0;
bool displayEnabled = true;
uint recordInterval = 0;
uint logInterval = 0;
uint partlogInt = 0;
bool bPause = false;
bool displaySliders = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 300;

enum { M_VIEW = 0, M_MOVE };

SimParams pdata;

// simulation parameters
float timestep = 500; //in units of nanoseconds
double simtime = 0.0f;
float externalH = 100; //kA/m
float colorFmax = 3.5;
float iter_dxpct = 0.035;
float rebuild_pct = 0.1;

float contact_dist = 1.05f;
float pin_dist = 1.05f;
float3 worldSize;
int numIterations = 0; // run until exit
float maxtime = 0;



int resolved = 0;//number of times the integrator had to resolve a step

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

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
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

void cleanup()
{
    cutilCheckError( cutDeleteTimer( timer));

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}
void setParams(){
    psystem->setGlobalDamping(pdata.globalDamping);
    psystem->setRepelSpring(pdata.spring);
    psystem->setShear(pdata.shear);
    psystem->setExternalH(make_float3(0.0f, externalH*1e3, 0.0f));
	psystem->setColorFmax(colorFmax*1e-7f);
	psystem->setViscosity(pdata.viscosity);
	psystem->setDipIt(pdata.mutDipIter);
	psystem->setPinDist(pin_dist);
	psystem->setContactDist(contact_dist);
	psystem->setRebuildDist(rebuild_pct);
}


void qupdate()
{
 	setParams();
	//this crude hack makes pinned particles at start unpinned so they can space and unfuck each other
	if(simtime < timestep)	psystem->setPinDist(0.8f);

	float dtout = 1e9*psystem->update(timestep*1e-9f, iter_dxpct);
	if(fabs(dtout - timestep) > .01f*dtout)
		resolved++;
	if(logInterval != 0 && frameCount % logInterval == 0){
		printf("iter %d at %.2f/%.1f us\n", frameCount, simtime*1e-3f, maxtime);
		psystem->logStuff(datalog, simtime*1e-3);	
	}

	if(frameCount % 1000 == 0){
		fprintf(crashlog, "Time: %g ns\n", simtime);
		psystem->logParams(crashlog);
		psystem->logParticles(crashlog);
		rewind(crashlog);//sets up the overwrite
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

    glutReportErrors();
}

void runBenchmark()
{
    printf("Run %u particles simulation for %f us...\n\n", pdata.numBodies, maxtime);
    cudaThreadSynchronize();
    cutilCheckError(cutStartTimer(timer));  
    //for (int ii = 0; ii < numIterations; ii++){
	while(simtime < maxtime*1e3f){
		qupdate();	    
	}
	//do a final write to the logfile
	if(logInterval > 0)
		psystem->logStuff(datalog, simtime*1e-3f);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer));  
    float fAvgSeconds = (1.0e-3 * cutGetTimerValue(timer))/numIterations;

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5fs, Size = %u particles\n", 
            (1.0e-3 * pdata.numBodies)/fAvgSeconds, fAvgSeconds*numIterations, pdata.numBodies);

}

void computeFPS()
{
    fpsCount++;
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "CUDA MR Fluid Sim (%d particles): %3.1f fps Time: %.2f us",pdata.numBodies, ifps, simtime*1e-3);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 

        cutilCheckError(cutResetTimer(timer));  
    }
}

void display()
{
  
  	cutilCheckError(cutStartTimer(timer));  
	setParams();
    // update the simulation
    if (!bPause)
    {
		qupdate();
		if (renderer){ 
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
		}
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(0.0, 0.0, 0.0);
    glutWireCube(worldSize.x);

    if (renderer && displayEnabled)
    {
        renderer->display(displayMode);
    }
	
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
        paramlist->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    cutilCheckError(cutStopTimer(timer));  
	

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
	if((numIterations > 0) && ((int)frameCount > numIterations)){
		printf("hi\n");
		exit(0);
	}
	if((maxtime > 0) && (simtime >= maxtime*1e3)){
		printf("yo\n");
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
    gluPerspective(60.0, (float) w / (float) h, 0.1e-6, 1);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer) {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
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

	uint nedges = 0, ngraphs = 0;
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
        if(g_useGL) {
			printf("the impending segfault appears to be a graphics issue, and not a code error\n");
		}
		exit(0);
        break;
    case 'v':
        mode = M_VIEW;
        break;
    case 'm':
        psystem->getMagnetization();
		break;
    case 'p':
        displayMode = (ParticleRenderer::DisplayMode)
                      ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
        break;
	case 'i':
		printf("interactions: %d\n", psystem->getInteractions());
		break;
	case 'n':
		psystem->NListStats();
		break;
	case 'd':
        psystem->dumpGrid();
        break;
	case 's': 
		psystem->printStress();
		break;
	case 'g':
		psystem->getGraphData(ngraphs,nedges);
		printf("numgraphs: %d, chainl: %f\n", ngraphs, (float)pdata.numBodies/(float) ngraphs);
		printf("Edges = %d Mean edges = %f\n", nedges, (float)nedges/(float)pdata.numBodies);
		break;
    case 'u':
        psystem->dumpParticles(0, pdata.numBodies);
        break;
    case 'r':
        displayEnabled = !displayEnabled;
        break;
	case '1':
        psystem->reset(1100, 0.3f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '2':
        psystem->reset(300, 1.0f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '3':
        psystem->reset(20, 1.0f);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '4':
        {
            // shoot ball from camera
            float vel[4], velw[4], pos[4], posw[4];
            vel[0] = 0.0f;
            vel[1] = 0.0f;
            vel[2] = -0.05f;
            vel[3] = 0.0f;
            ixform(vel, velw, modelView);

            pos[0] = 0.0f;
            pos[1] = 0.0f;
            pos[2] = -2.5f;
            pos[3] = 1.0;
            ixformPoint(pos, posw, modelView);
            posw[3] = 0.0f;

        }
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

    if (demoMode) {
        camera_rot[1] += 0.02f;
        /*if (demoCounter++ > 1000) {
            ballr = 10 + (rand() % 10);
            demoCounter = 0;
        }*/
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
		paramlist->AddParam(new Param<float>("colorFmax", colorFmax, 0, 15, 0.1f, &colorFmax));
    	paramlist->AddParam(new Param<float>("visc", pdata.viscosity, 0.001f, .25f, 0.001f, &pdata.viscosity));
		paramlist->AddParam(new Param<float>("max dx pct", iter_dxpct, 0, .2f, 0.002f, &iter_dxpct));
		paramlist->AddParam(new Param<float>("pin dist", pin_dist, 0.995f, 1.5f, 0.005f, &pin_dist));
		paramlist->AddParam(new Param<float>("contact_dist", contact_dist, .95f, 1.25f, 0.001f, &contact_dist));
		paramlist->AddParam(new Param<float>("rebuild dist", rebuild_pct, 0.0f, 1.0f, 0.005f, &rebuild_pct));
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
    shrSetLogFileName ("particles.txt");
    shrLog("%s Starting...\n\n", argv[0]);


	numIterations = 0;
	int devID = cutGetMaxGflopsDeviceId();
	if(cutCheckCmdLineFlag(argc, (const char **)argv, "noGL")) {
		g_useGL = false;
		cutGetCmdLineArgumenti(argc, (const char**) argv, "device", (int *) &devID);
	}
	cudaSetDevice(devID);
	printf("devID: %d\n", devID);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "record", (int *) &recordInterval);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "logf", (int *) &logInterval);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "plogf", (int *) &partlogInt);	
	//set the random seed
	uint randseed = time(NULL);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "randseed", (int *) &randseed);
	srand(randseed);
	printf("randseed: %d\n", randseed);
	
	
	
	//DEFINE SIMULATION PARAMETERS

	float worldsize1d = .35;//units of mm
	cutGetCmdLineArgumentf(argc, (const char**)argv, "wsize", (float*) &worldsize1d);
	worldSize = make_float3(worldsize1d*1e-3f, worldsize1d*1e-3f, worldsize1d*1e-3f);
	float volume = worldSize.x*worldSize.y*worldSize.z; 

	pdata.shear = 500;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "shear", (float*) &pdata.shear);
	
	pdata.spring = 50;	
	cutGetCmdLineArgumentf(argc, (const char**)argv, "k", (float*) &pdata.spring);

	externalH = 100;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "H", (float*) &externalH);
	pdata.externalH = make_float3(0, externalH*1e3f, 0);

	cutGetCmdLineArgumentf(argc, (const char**)argv, "dt", (float*) &timestep);//units of ns
	cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &numIterations);
	maxtime =timestep*numIterations*1e-3;//maxtime has units of us for simplicity
	cutGetCmdLineArgumentf(argc, (const char**) argv, "maxtime", &maxtime);//units of ns as well
	numIterations = maxtime/timestep*1e3f;
	pdata.viscosity = 0.1;
	cutGetCmdLineArgumentf(argc, (const char**)argc, "visc", (float*) &pdata.viscosity);

	pdata.colorFmax = colorFmax*1e-7;
	pdata.globalDamping = 0.8f; 
	pdata.cdamping = 0.03f;
	pdata.cspring = 10;
	cutGetCmdLineArgumentf(argc, (const char**)argc, "cspring", (float*)&pdata.cspring);
	pdata.boundaryDamping = -0.03f;

	cutGetCmdLineArgumentf(argc, (const char**)argc, "pin_d", (float*)&pin_dist);
	cutGetCmdLineArgumentf(argc, (const char**)argc, "contact_dist", (float*)&contact_dist);
	cutGetCmdLineArgumentf(argc, (const char**)argc, "rebuild_dist", (float*)&rebuild_pct);
	cutGetCmdLineArgumentf(argc, (const char**)argc, "iterdx",(float*)&iter_dxpct); 
		
	pdata.mutDipIter = 0;
	cutGetCmdLineArgumenti(argc, (const char**)argc, "dipit", (int*) &pdata.mutDipIter);
		

	char* title;	
	if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "logt", &title)){
		filename = title;
	}
	
   	sprintf(logfile, "/home/steve/Datasets/%s.dat", filename);
	sprintf(crashname, "/home/steve/Datasets/%s.crash.dat", filename);

	pdata.flowmode = false;
	if(cutCheckCmdLineFlag(argc, (const char **)argv, "flowmode")) {	
		pdata.flowmode = true;
	}
	pdata.flowvel = 2e7; 
	pdata.nd_plug = .20;	

	float radius = 4.0f;
	//haven't figured out a good way to do this in a loop
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad0", (float*) &radius);
	pdata.pRadius[0] = radius*1e-6f; //median diameter
	pdata.volfr[0] = 0.30f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr0", (float*) &pdata.volfr[0]);
	pdata.mu_p[0] = 2000; //relative permeability
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi0", (float*) &pdata.mu_p[0]);
	pdata.nump[0] = (pdata.volfr[0] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[0],3)); 
	pdata.rstd[0] = 0; //sigma0 in log normal distribution
	cutGetCmdLineArgumentf(argc, (const char**)argv, "std0", (float*) &pdata.rstd[0]);	
	if(pdata.rstd[0] > 0){//eq 3.24 crowe
		pdata.nump[0] = (pdata.volfr[0]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[0],3)
					*exp(4.5f*pdata.rstd[0]*pdata.rstd[0]));
	}

	radius = 6.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad1", (float*) &radius);
	pdata.pRadius[1] = radius*1e-6f;
	pdata.volfr[1] = 0.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr1", (float*) &pdata.volfr[1]);
	pdata.mu_p[1] = 2000;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi1", (float*) &pdata.mu_p[1]);
	pdata.nump[1] = (pdata.volfr[1] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[1],3)); 
	pdata.rstd[1] = 0;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "std1", (float*) &pdata.rstd[1]);
	if(pdata.rstd[1] > 0){//eq 3.24 crowe
		pdata.nump[1] = (pdata.volfr[1]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[1],3)
					*exp(4.5f*pdata.rstd[1]*pdata.rstd[1]));
	}
	
	radius = 25.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad2", (float*) &radius);
	pdata.pRadius[2] = radius*1e-6f;
	pdata.volfr[2] = 0.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr2", (float*) &pdata.volfr[2]);
	pdata.mu_p[2] = 1;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi2", (float*) &pdata.mu_p[2]);
	pdata.nump[2] = (pdata.volfr[2] * volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[2],3)); 
	pdata.rstd[2] = 0;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "std2", (float*) &pdata.rstd[2]);
	if(pdata.rstd[2] > 0){//eq 3.24 crowe
		pdata.nump[2] = (pdata.volfr[2]*volume) / (4.0f/3.0f*PI_F*pow(pdata.pRadius[2],3)
					*exp(4.5f*pdata.rstd[2]*pdata.rstd[2]));
	}

	pdata.numBodies = pdata.nump[0] + pdata.nump[1] + pdata.nump[2];
	bool benchmark = cutCheckCmdLineFlag(argc, (const char**) argv, "benchmark") != 0;


	pdata.worldOrigin = worldSize*-0.5f;
	float cellSize_des = 8.0f*pdata.pRadius[0];
 
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

    cutilCheckError(cutCreateTimer(&timer));

	initParamList();
	setParams();
	psystem->reset(11, 0.4f);
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
		psystem->logParams(datalog);	
		fprintf(datalog, "time\tshear\textH\tchainl\tedges\ttopf\tbotf\tgstress\tkinen\tM.x \tM.y \tM.z\n");
	}
	
	crashlog = fopen(crashname, "w");


    if (benchmark || !g_useGL) 
    {
        if (numIterations <= 0){ 
			maxtime = timestep*500*1e-3;
			numIterations = 500;
		}
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

    shrEXIT(argc, (const char**)argv);
}
