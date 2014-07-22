void cleanup();
void setParams();
void qupdate();
void initGL(int argc, char **argv);
void runBenchmark();
void computeFPS();
void display();
inline float frand();
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void xform(float *v, float *r, GLfloat *m);
void ixform(float *v, float *r, GLfloat *m);
void ixformPoint(float *v, float *r, GLfloat *m);
void motion(int x, int y);
void key(unsigned char key, int /*x*/, int /*y*/);
void special(int k, int x, int y);
void idle(void);
void initParamList();
void mainMenu(int i);
void initMenus();
int main(int argc, char** argv) ;