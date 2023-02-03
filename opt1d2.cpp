#include<cstdlib>

#include<numpy.h>

namespace py = pybind11;
using namespace std;

#ifdef VERBOSE
	#include<cstdio>
	#define eprintf(...) printf(__VA_ARGS__);
#else
	#define eprintf(...);
#endif


double getCost(double a, double b) {
	return pow(a-b,2);
}

void getMinIndex(double *x, double *y, double *psi, size_t i, size_t jLast, bool firstAssigned, size_t &j, double &valj, size_t M, size_t N) {
	size_t jmin=0;
	if(firstAssigned) {
		// if some column is already assigned, can start at least there, no need to go back further
		jmin=jLast;
	}
	double valmin=getCost(x[i],y[jmin])-psi[jmin];
	double val;
	for(size_t jcur=jmin+1;jcur<N;jcur++) {
		val=getCost(x[i],y[jcur])-psi[jcur];
		if (val<valmin) {
			valmin=val;
			jmin=jcur;
		} else {
		// experimental: hunch: if this is false, then it won't get better later on, can abort here and return right away
			j=jmin;
			valj=valmin;
			return;
		}
	}
	j=jmin;
	valj=valmin;
}

void solve(double *x, double *y, double *phi, double *psi, int *piRow, int *piCol, double lam, size_t M, size_t N) {
	size_t K=0;
	bool firstAssigned=false; // whether at least one column has already been assigned
	size_t jLast=0; // highest currently assigned col index (if firstAssigned==true)
	
	double *dist=(double*) malloc(sizeof(double)*M);
	// experimental: this might not be needed, will always be initialized "on demand", see further down
//	for(size_t i=0;i<M;i++) {
//		dist[i]=INFINITY;
//	}
	
	while(K<M) {
		size_t j;
		double val;
		getMinIndex(x,y,psi,K,jLast,firstAssigned,j,val,M,N);
		if (val>=lam) {
			// eprintf("case 1\n");
			phi[K]=lam;
			K++;
		} else {
			if (piCol[j]==-1) {
				// eprintf("case 2\n");
				piCol[j]=K;
				piRow[K]=j;
				phi[K]=val;
				K++;
				jLast=j;
				firstAssigned=true;
			} else {
				// eprintf("case 3\n");
				phi[K]=val;
				// reset distance array for Dijkstra
				// experimental: probably this can also be simplified, I guess, to only reset this on entries where it is needed
				// maybe we don't even need this because it is not used for finding the next best edge
//				for(size_t i=0;i<M;i++) {
//					dist[i]=INFINITY;
//				}
				dist[K]=0.;
				dist[K-1]=0.;
				double v=0;
				// iMin and jMin indicate lower end of range of contiguous rows and cols
				// that are currently examined in subroutine;
				// upper end is always K and j
				size_t iMin=K-1;
				size_t jMin=j;
				// threshold until an entry of phi hits lam
				size_t lamInd;
				double lamDiff, lowEndDiff, hiEndDiff;
				if (phi[K]>phi[K-1]) {
					lamDiff=lam-phi[K];
					lamInd=K;
				} else {
					lamDiff=lam-phi[K-1];
					lamInd=K-1;
				}
				bool resolved=false;
				while(!resolved) {
					//threshold until constr iMin,jMin-1 becomes active
					if (jMin>0) {
						lowEndDiff=getCost(x[iMin],y[jMin-1])-phi[iMin]-psi[jMin-1];
					} else {
						lowEndDiff=INFINITY;
					}
			                // threshold for upper end
					if (j<N-1) {
						hiEndDiff=getCost(x[K],y[j+1])-phi[K]-psi[j+1]-v;
					} else {
						hiEndDiff=INFINITY;
					}
					if ((hiEndDiff<=lowEndDiff) && (hiEndDiff<=lamDiff)) {
						// eprintf("case 3.1\n");
						v+=hiEndDiff;
						for(size_t i=iMin;i<K;i++) {
							phi[i]+=v-dist[i];
							psi[piRow[i]]-=v-dist[i];
						}
						phi[K]+=v;
						piRow[K]=j+1;
						piCol[j+1]=K;
						resolved=true;
						jLast=j+1;
						firstAssigned=true;
					} else {
						if ((lowEndDiff<=hiEndDiff) && (lowEndDiff<=lamDiff)) {
							if (piCol[jMin-1]==-1) {
								// eprintf("case 3.2a\n");
								v+=lowEndDiff;
								for (size_t i=iMin;i<K;i++) {
									phi[i]+=v-dist[i];
									psi[piRow[i]]-=v-dist[i];
								}
								phi[K]+=v;
								// "flip" assignment along whole chain
								size_t jPrime=jMin;
								piCol[jMin-1]=iMin;
								piRow[iMin]-=1;
								for(size_t i=iMin+1;i<K;i++) {
									piCol[jPrime]+=1;
									piRow[i]-=1;
									jPrime+=1;
								}
								piRow[K]=jPrime;
								piCol[jPrime]+=1;
								resolved=true;

							} else {
								// eprintf("case 3.2b\n");
								v+=lowEndDiff;
								dist[iMin-1]=v;
								// adjust distance to threshold
								lamDiff-=lowEndDiff;
								iMin-=1;
								jMin-=1;
								if (lam-phi[iMin]<lamDiff) {
									lamDiff=lam-phi[iMin];
									lamInd=iMin;
								}
							}
						} else {
							// eprintf("case 3.3, lamInd=%lu",lamInd);
							v+=lamDiff;
							for(size_t i=iMin;i<K;i++) {
								phi[i]+=v-dist[i];
								psi[piRow[i]]-=v-dist[i];
							}
							phi[K]+=v;
							// "flip" assignment from lambda touching row onwards
							size_t jPrime=piRow[lamInd];
							piRow[lamInd]=-1;
							for(size_t i=lamInd+1;i<K;i++) {
								piCol[jPrime]+=1;
								piRow[i]-=1;
								jPrime+=1;
							}
							if (lamInd<K) {
								piRow[K]=jPrime;
								piCol[jPrime]+=1;
							}
							resolved=true;
						} // end case 3.3 (=case 3.2 else)
					} // end case 3.1 else
                    		} // end case 3 subroutine while
				K++;
			} // end case 3 (=case 2 else)
		} // end case 1: else
	} // end K loop
	
	free(dist);
}
