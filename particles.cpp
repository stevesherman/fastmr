


#define PI 3.141592653589793f

// Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

//CUDA utilities and system inlcudes
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <rendercheck_gl.h>
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

//shared library test functions
#include <shrUtils.h>
#include <shrQATest.h>

#define MV_AVG_SIZE 30 
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
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 300;

enum { M_VIEW = 0, M_MOVE };


SimParams params;

uint numParticles = 0;
uint3 gridSize;
float3 worldSize;
int numIterations = 0; // run until exit
float maxtime = 0;

// simulation parameters
float timestep = 1000; //in units of nanoseconds
double simtime = 0.0f;
float externalH = 100; //kA/m
float colorFmax = 3.5;
float maxdxpct = 0.10;
float resolvetime [MV_AVG_SIZE];


float old_dt [MV_AVG_SIZE];
float lost_time_old, lost_time_new;
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

const char *sSDKsample = "CUDA MR Fluid Simulation";

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
    psystem->setGlobalDamping(params.globalDamping);
    psystem->setRepelSpring(params.spring);
    psystem->setCollideDamping(params.cdamping);
    psystem->setShear(params.shear);
	psystem->setInteractionRadius(params.interactionr);
    psystem->setExternalH(make_float3(0.0f, externalH*1e3, 0.0f));
	psystem->setColorFmax(colorFmax*1e-7f);
	psystem->setViscosity(params.viscosity);
	psystem->setDipIt(params.mutDipIter);
}


void qupdate()
{
 	setParams();
	float dtout = 1e9*psystem->update(timestep*1e-9f, maxdxpct);
	if(fabs(dtout - timestep) > .01f*dtout)
		resolved++;

		
/*	
	old_dt[frameCount % MV_AVG_SIZE] = dtout;
	//printf("dtout: %g, timestep %g\n", dtout, timestep);
	if(fabs(dtout - timestep) > .01f*dtout){
		resolved++;
		resolvetime[frameCount % MV_AVG_SIZE] = timestep + timestep - dtout;
	} else {
	   	printf("hi");	

		resolvetime[frameCount % MV_AVG_SIZE] = 0;
	}	
	const float incrate = 0.05f;

	float resolvetotalt = 0;
	float dt_total = 0;
	for(ii = 0; ii < MV_AVG_SIZE; ii++){
		resolvetotalt += resolvetime[ii];
		dt_total += old_dt[ii];
	}


	//incrate*dt_total is how much savings we expect
	//resolve_totalt is the time we spent resolving
	printf("timesaved by incing: %g \tdt: %f\t timespent resolving %g\n", dt_total*incrate, timestep, resolvetotalt);
	if(frameCount > 300){
		if(dt_total*incrate >  2*resolvetotalt)
			timestep *= 1 + incrate;
		if(dt_total*incrate < 0.5*resolvetotalt)		
			timestep *= 1 - incrate;
	}*/
	//printf("timestep: %g\n", timestep);
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
		fprintf(crashlog, "Time: %g ns\n", simtime);
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
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void runBenchmark()
{
    printf("Run %u particles simulation for %f us...\n\n", params.numBodies, maxtime);
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
            (1.0e-3 * params.numBodies)/fAvgSeconds, fAvgSeconds*numIterations, params.numBodies);

}

void computeFPS()
{
    fpsCount++;
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "CUDA MR Fluid Sim (%d particles): %3.1f fps Time: %.2f us",params.numBodies, ifps, simtime*1e-3);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 

        cutilCheckError(cutResetTimer(timer));  
    }
}

void display()
{
  
  	cutilCheckError(cutStartTimer(timer));  

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
    //glColor3f(1.0, 1.0, 1.0);
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

	float ne = 0;
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
    case 'd':
        psystem->dumpGrid();
        break;
	case 'g':
		printf("numgraphs: %d, chainl: %f\n", psystem->getGraphs(), (float)params.numBodies/(float)psystem->getGraphs());
		break;
    case 'u':
        psystem->dumpParticles(0, params.numBodies);
        break;
	case 'c':
		ne = psystem->getEdges();
		printf("Edges = %f\n", ne);
		break;
    case 'r':
        displayEnabled = !displayEnabled;
        break;

    case '1':
        psystem->reset(ParticleSystem::CONFIG_GRID);
		frameCount = 0; simtime = 0; resolved = 0;
        break;
    case '2':
        psystem->reset(ParticleSystem::CONFIG_RANDOM);
		frameCount=0; simtime = 0; resolved = 0;
		break;
    case '3':
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

    case 'w':
        wireframe = !wireframe;
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
		paramlist->AddParam(new Param<float>("spring constant",params.spring, 0, 100, 1, &params.spring));
		paramlist->AddParam(new Param<int>("interaction radius", params.interactionr, 1,10,1,&params.interactionr));
		paramlist->AddParam(new Param<float>("H (kA/m)", externalH, 0, 1e3, 5, &externalH));
		paramlist->AddParam(new Param<float>("shear rate", params.shear, 0, 2000, 50, &params.shear));
		paramlist->AddParam(new Param<float>("colorFmax", colorFmax, 0, 15, 0.1f, &colorFmax));
    	paramlist->AddParam(new Param<float>("visc", params.viscosity, 0.001f, .25f, 0.001f, &params.viscosity));
		paramlist->AddParam(new Param<int>("DipIter", params.mutDipIter, 0, 5, 1, &params.mutDipIter));	
		paramlist->AddParam(new Param<float>("max dx pct", maxdxpct, 0, .5f, 0.005f, &maxdxpct));
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
	cudaSetDevice(devID);
	printf("devID: %d\n", devID);
	if(cutCheckCmdLineFlag(argc, (const char **)argv, "noGL"))
			g_useGL = false;
	cutGetCmdLineArgumenti(argc, (const char **)argv, "record", (int *) &recordInterval);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "logf", (int *) &logInterval);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "plogf", (int *) &partlogInt);	
	//set the random seed
	uint randseed = time(NULL);
	cutGetCmdLineArgumenti(argc, (const char **)argv, "randseed", (int *) &randseed);
	srand(randseed);
	printf("randseed: %d\n", randseed);
	
	
	
	//DEFINE SIMULATION PARAMETERS

	float worldsize1d = .224;//units of mm
	cutGetCmdLineArgumentf(argc, (const char**)argv, "wsize", (float*) &worldsize1d);
	worldSize = make_float3(worldsize1d*1e-3f, worldsize1d*1e-3f, worldsize1d*1e-3f);
	params.worldSize = worldSize;
	float volume = worldSize.x*worldSize.y*worldSize.z; 

	params.shear = 500;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "shear", (float*) &params.shear);
	
	params.spring = 25;	
	cutGetCmdLineArgumentf(argc, (const char**)argv, "k", (float*) &params.spring);

	externalH = 100;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "H", (float*) &externalH);
	params.externalH = make_float3(0, externalH, 0);

	cutGetCmdLineArgumentf(argc, (const char**)argv, "dt", (float*) &timestep);//units of ns
	cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &numIterations);
	maxtime =timestep*numIterations*1e-3;//maxtime has units of us for simplicity
	cutGetCmdLineArgumentf(argc, (const char**) argv, "maxtime", &maxtime);//units of ns as well
	numIterations = maxtime/timestep*1e3f;
	params.viscosity = 0.1;
	cutGetCmdLineArgumentf(argc, (const char**)argc, "visc", (float*) &params.viscosity);

	params.colorFmax = colorFmax*1e-7;
 	params.interactionr = 2;
	params.globalDamping = 0.8f; 
	params.cdamping = 0.03f;
	params.cspring = 10;
	cutGetCmdLineArgumentf(argc, (const char**)argc, "cspring", (float*)&params.cspring);
	params.boundaryDamping = -0.03f;

	params.randSetIter = 300;
	cutGetCmdLineArgumenti(argc, (const char**)argc, "randit", (int*)&params.randSetIter);
		
	params.mutDipIter = 0;
	cutGetCmdLineArgumenti(argc, (const char**)argc, "dipit", (int*) &params.mutDipIter);


	char* title;	
	if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "logt", &title)){
		filename = title;
	}
	
   	sprintf(logfile, "/home/steve/Datasets/%s.dat", filename);
	sprintf(crashname, "/home/steve/Datasets/%s.crash.dat", filename);

	params.flowmode = false;
	if(cutCheckCmdLineFlag(argc, (const char **)argv, "flowmode")) {	
		params.flowmode = true;
	}
	params.flowvel = 2e7; 
	params.nd_plug = .20;	

	float radius = 3.5f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad0", (float*) &radius);
	params.particleRadius[0] = radius*1e-6f;
	params.volfr[0] = 0.30f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr0", (float*) &params.volfr[0]);
	params.xi[0] = 2000;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi0", (float*) &params.volfr[0]);
	params.numParticles[0] = (params.volfr[0] * volume) / (4.0f/3.0f*PI*pow(params.particleRadius[0],3)); 
		
	radius = 6.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad1", (float*) &radius);
	params.particleRadius[1] = radius*1e-6f;
	params.volfr[1] = 0.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr1", (float*) &params.volfr[1]);
	params.xi[1] = 1;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi1", (float*) &params.volfr[1]);
	params.numParticles[1] = (params.volfr[1] * volume) / (4.0f/3.0f*PI*pow(params.particleRadius[1],3)); 
	
	radius = 3.5f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "rad2", (float*) &radius);
	params.particleRadius[2] = radius*1e-6f;
	params.volfr[2] = 0.0f;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "vfr2", (float*) &params.volfr[2]);
	params.xi[2] = 2000;
	cutGetCmdLineArgumentf(argc, (const char**)argv, "xi2", (float*) &params.volfr[2]);
	params.numParticles[2] = (params.volfr[2] * volume) / (4.0f/3.0f*PI*pow(params.particleRadius[2],3)); 

	params.numBodies = params.numParticles[0] + params.numParticles[1] + params.numParticles[2];
    bool benchmark = cutCheckCmdLineFlag(argc, (const char**) argv, "benchmark") != 0;


	if(cutGetCmdLineArgumentstr( argc, (const char**)argv, "crash", &title )){//some of crash flag
		char xyz[256];
	   	sprintf(xyz, "/home/steve/Datasets/%s", title);
		printf("reading crash file:  %s\n",xyz);
		crashlog = fopen(xyz, "r");
		if(crashlog == NULL){	
			printf("Crash file failed to open!");
		} else {
			int a = fscanf(crashlog, "Time: %lg ns\n", &simtime);
			if( a == 0) rewind(crashlog);	
			a = fscanf(crashlog, "vfrtot: %*f\t v0: %f\t v1: %f\t v2: %f\n", &params.volfr[0], 
					&params.volfr[1], &params.volfr[2]);
			a = fscanf(crashlog, "ntotal: %d\t n0: %d  \t n1: %d  \t n2: %d\n", &params.numBodies, &params.numParticles[0],
					&params.numParticles[1], &params.numParticles[2]);
			a = fscanf(crashlog, "\t\t xi0: %f \t xi1: %f \t xi2 %f \n", &params.xi[0], &params.xi[1], &params.xi[2]);
			//printf("xis read: %d\n", a);
			a = fscanf(crashlog, "\t\t a0: %g\t a1: %g\t a2: %g\n\n", &params.particleRadius[0], &params.particleRadius[1],
					&params.particleRadius[2]);
			printf("rads read: %d\n", a);
			//a = fscanf(crashlog, "grid: %d x %d x %d = %*d cells\n", &gridSize.x, &gridSize.y, &gridSize.z);
			printf("grid read: %d grid: %d x %d x %d\n", a, params.gridSize.x, params.gridSize.y, params.gridSize.z);
			//a = fscanf(crashlog, "worldsize: %fmm x %*fmm x %*fmm\n", &worldsize1d);
			printf("wsize read: %d, wsize: %.3f\n", a, worldsize1d);
			//a = fscanf(crashlog, "\nspring: %f\tvisc: %f\tdipit: %d\n", &params.spring, &params.viscosity, &params.mutDipIter);
			printf("params read: %d\n", a);
			//a = fscanf(crashlog, "H.x: %g\tH.y: %g\tH.z: %g\n", &params.externalH.x, &params.externalH.y, &params.externalH.z);
			printf("H read: %d\n", a);
			fclose(crashlog);
		}
	}

	params.worldOrigin = worldSize*-0.5f;
	float cellSize = 8.0f*params.particleRadius[0];
	params.cellSize = make_float3(cellSize, cellSize, cellSize);	
 
	//gridSize.x = gridSize.y = gridSize.z = GRID_SIZE;
	if(fmod(worldSize.x , cellSize) < (.1f*params.particleRadius[0])){
		params.gridSize.x = floor(worldSize.x/cellSize);
	} else {
		params.gridSize.x = ceil(worldSize.x/cellSize);
	}
	if(fmod(worldSize.y , cellSize) < (.1f*params.particleRadius[0])){
		params.gridSize.y = floor(worldSize.y/cellSize);
	} else {
		params.gridSize.y = ceil(worldSize.y/cellSize);
	}
	if(fmod(worldSize.z , cellSize) < (.1f*params.particleRadius[0])){
		params.gridSize.z = floor(worldSize.z/cellSize);
	} else {
		params.gridSize.z = ceil(worldSize.z/cellSize);
	}


	//INITIALIZE THE DATA STRUCTURES

    if (!g_useGL) {
        cudaInit(argc, argv);
    } else {
        initGL(argc, argv);
        cudaGLInit(argc, argv);
    }
	psystem = new ParticleSystem(params, g_useGL, worldSize);
	psystem->logParams(stdout); 
	if (g_useGL) {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    cutilCheckError(cutCreateTimer(&timer));

	initParamList();
	setParams();
	psystem->reset(ParticleSystem::CONFIG_RANDOM);
    if (g_useGL) 
        initMenus();

	if( recordInterval != 0 ) {
        g_CheckRender = new CheckBackBuffer(width, height, 4);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }

	if(logInterval != 0){
		printf("saving: %s\n",logfile);
		datalog = fopen(logfile, "a");
		psystem->logParams(datalog);	
		fprintf(datalog, "time\tshear\textH\tchainl\tedges\ttopf\tbotf\tgstress\tkinen\tM.x \tM.y \tM.z\n");
	}
	
	crashlog = fopen(crashname, "w");

	for(int ii=0; ii < MV_AVG_SIZE; ii++){
		resolvetime[ii]=0;
		old_dt[ii] = 0;
	}	

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
