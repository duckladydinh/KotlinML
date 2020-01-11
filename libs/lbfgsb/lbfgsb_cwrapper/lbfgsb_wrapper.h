#ifndef LBFGSB_WRAPPER
#define LBFGSB_WRAPPER

#define LBFGSB_TASK_SIZE 60

enum TaskType { LBFGSB_FG, LBFGSB_NEW_X, LBFGSB_CONV, LBFGSB_ABNO,
	LBFGSB_ERROR, LBFGSB_START, LBFGSB_STOP, LBFGSB_UNKNOWN };

// Data used by the algorithm. 
// Check the original Fortran source code to see what each variable means.
struct L_BFGS_B {
	int n; // variables no.
	int m; // corrections no.
	double* x; // point
	double* l; // lower bounds
	double* u; // upper bounds
	int* nbd; // bound type for each dimension: 
				// 0 - unbounded,
				// 1 - only a lower bound,
				// 2 - both lower and upper bounds, 
				// 3 - only an upper bound.
	double f; // value of the function at point x
	double* g; // gradient of the function at point x
	double factr; // function value stop condition tolerance factor, 0 to suppress this stop condition
	double pgtol; // gradient norm value stop condition
	double* wa; // workspace array 
	int* iwa; // integer workspace array
	char task[LBFGSB_TASK_SIZE+1]; // task name string:
					// Strings returned by the algorithm:
					// "FG"
					// "NEW_X"
					// "CONV"
					// "ABNO"
					// "ERROR"
					// Strings that can be set by the user:
					// "START"
					// "STOP"
	int iprint; // output printing frequency 
	char csave[60]; // character working array
	int lsave[4]; // logical working array
	int isave[44]; // integer working array
	double dsave[29]; // double working array
};

// Create algorithm data with default values set where it is possible
struct L_BFGS_B* create(int n, int m);

// Delete algorithm structure
void close(struct L_BFGS_B* data);

void step(struct L_BFGS_B* data);

void set_task(struct L_BFGS_B* data, enum TaskType type);

enum TaskType get_task(struct L_BFGS_B* data);

// Internal task manipulation function: 
// Copies C-string (with '\0' at the end), to Fortran-string (padded with spaces)
void set_task_str(struct L_BFGS_B* data, char* c_str);

// Internal task manipulation function: 
// @return true iff the task description is equal up to the length of c_str
int is_task_str_equal(struct L_BFGS_B* data, char* c_str);

#endif
