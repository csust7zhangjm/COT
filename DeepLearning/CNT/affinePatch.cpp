# include<math.h>
# include "mex.h"

#define Wimgs      prhs[0]
#define Patchsize  prhs[1]
#define Patchnum   prhs[2]

#define ii_OUT     plhs[0]

void mexFunction(int nlhs, mxArray * plhs[],
	             int nrhs, const mxArray*prhs[])
{
	int i;
	int j;
	int k;
	int l;
	int m;
	int c;

	double* patchsize;
	double* patchnum;
	double* wimgs;

	patchsize = mxGetPr(Patchsize);
	patchnum = mxGetPr(Patchnum);
	wimgs = mxGetPr(Wimgs);

 
	double* ii_out;

  
	const mwSize* dim3s = mxGetDimensions(Wimgs);
	
	int blocksizeHeight = int(dim3s[0]);
	int blocksizeWidth  = int(dim3s[1]);
	int n = int(dim3s[2]);
    
	int y = int(patchsize[0]/2);
	int x = int(patchsize[1]/2);

    int patch_cent_leny = blocksizeHeight - 2*y + 1;
	int patch_cent_lenx = blocksizeWidth - 2*x + 1;

	int *patch_centy = new int(patch_cent_leny);
	int *patch_centx = new int(patch_cent_lenx);

	for(k=0; k<patch_cent_leny; k++)
	{
	    patch_centy[k] = y;
		patch_centx[k] = x;

		y = y + 1;
		x = x + 1;
	}
    
     //--------  creat output matrix ----------
	int dimout[3];
	dimout[0]=int(patchsize[0]*patchsize[1]);
	dimout[1] = int(patchnum[0]*patchnum[1]);
	dimout[2] = n;
    ii_OUT=mxCreateNumericArray(3,dimout,mxDOUBLE_CLASS,mxREAL);
	ii_out=mxGetPr(ii_OUT);

	//for(i=0;i<dimout[0]*dimout[1]*dimout[2];i++)
	//	ii_out[i] = i;
	//
    c = 0;
	for(i=0;i<n;i++)
		for(j=0;j<patchnum[0];j++)
			for(k=0;k<patchnum[1];k++)
				for(l=0;l<patchsize[0];l++)
					for(m=0;m<patchsize[1];m++)
					{
                        //ii_out[c] = wimgs[blocksizeHeight*blocksizeWidth*i+(patch_centx[j]-int(patchsize[1]/2)+l)*blocksizeHeight+patch_centy[k]-int(patchsize[0]/2)+m];
						ii_out[c] =  blocksizeHeight*blocksizeWidth*i+(patch_centx[j]-int(patchsize[1]/2)+l)*blocksizeHeight+patch_centy[k]-int(patchsize[0]/2)+m;
						c = c + 1;
					}
	//
   

					delete []patch_centy;
					delete []patch_centx;
					
			
		
}