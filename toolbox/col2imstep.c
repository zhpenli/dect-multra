/**************************************************************************
 *
 *   Image = col2imstep(Patch, ImgSiz, PatSiz, SldDist)
 *   prhs[0] - patches  
 *   prhs[1] - image size
 *   prhs[2] - patch size
 *   prhs[3] - sliding distance
 *   plhs[0] - restored image
 *   
 *   Zhipeng Li, UM-SJTU Joint Institute
 *
 *************************************************************************/


#include "mex.h"


/* Input Arguments */

#define	B_IN	 prhs[0]
#define N_IN   prhs[1]
#define SZ_IN  prhs[2]
#define S_IN   prhs[3]


/* Output Arguments */

#define	X_OUT	plhs[0]


void mexFunction(int nlhs, mxArray *plhs[], 
		             int nrhs, const mxArray*prhs[])
     
{ 
    float *x, *b;
    double *s;
    mwSize sz[3], stepsize[3], n[3], ndims;
    mwIndex i, j, k, l, m, t, blocknum;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 3 || nrhs > 4) {
      mexErrMsgTxt("Invalid number of input arguments."); 
    } else if (nlhs > 1) {
      mexErrMsgTxt("Too many output arguments."); 
    } 
    
    
    /* Check the the input dimensions */ 
    
    if (!mxIsSingle(B_IN) ||  mxIsComplex(B_IN) || mxGetNumberOfDimensions(B_IN)>2) {
      mexErrMsgTxt("B should be a single matrix.");
    }
    if (!mxIsDouble(N_IN) ||  mxIsComplex(N_IN) || mxGetNumberOfDimensions(N_IN)>2) {
      mexErrMsgTxt("Invalid output matrix size.");
    }

    ndims = mxGetM(N_IN)*mxGetN(N_IN);


    if (ndims<2 || ndims>3) {
      mexErrMsgTxt("Output matrix can only be 2-D or 3-D.");
    }
    if (!mxIsDouble(SZ_IN) || mxIsComplex(SZ_IN) || mxGetNumberOfDimensions(SZ_IN)>2 || mxGetM(SZ_IN)*mxGetN(SZ_IN)!=ndims) {
      mexErrMsgTxt("Invalid block size.");
    }
    if (nrhs == 4) 
    {
      if ( !mxIsDouble(S_IN) || mxIsComplex(S_IN) || mxGetNumberOfDimensions(S_IN)>2 || mxGetM(S_IN)*mxGetN(S_IN)!=ndims) 
      {
        mexErrMsgTxt("Invalid step size.");
      }
    }
    
    
    /* Get parameters */
    
    s = mxGetPr(N_IN);
    if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) 
    {
      mexErrMsgTxt("Invalid output matrix size.");
    }
    n[0] = (mwSize)(s[0] + 0.01);
    n[1] = (mwSize)(s[1] + 0.01);
    n[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    s = mxGetPr(SZ_IN);
    if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) {
      mexErrMsgTxt("Invalid block size.");
    }
    sz[0] = (mwSize)(s[0] + 0.01);
    sz[1] = (mwSize)(s[1] + 0.01);
    sz[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 4)
    {
      s = mxGetPr(S_IN);
      if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) 
      {
        mexErrMsgTxt("Invalid step size.");
      }
      stepsize[0] = (mwSize)(s[0] + 0.01);
      stepsize[1] = (mwSize)(s[1] + 0.01);
      stepsize[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    }
    else
    {
      stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
    if (n[0]<sz[0] || n[1]<sz[1] || (ndims==3 && n[2]<sz[2])) {
      mexErrMsgTxt("Block size too large.");
    }


    
    
    /* Create a matrix for the return argument */
    
    X_OUT = mxCreateNumericArray(ndims, n, mxSINGLE_CLASS, mxREAL);
    
    
    /* Assign pointers */
    
    b = (float *)mxGetPr(B_IN);
    x = (float *)mxGetPr(X_OUT);
            
    
    /* Do the actual computation */
    
    blocknum = 0;
    
    /* iterate over all blocks */
 for (k=0; k<=n[2]-sz[2]; k+=stepsize[2]) 
    {
      for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
      {
        for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
          
          /* add single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
          
        }
        if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
                for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
           }
      }
       if(j<n[1]-sz[1]+stepsize[1])
          {   
              j=n[1]-sz[1];
               for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
          
          /* add single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
          
        }
        if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
                for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
           }
          }
    }
     if(k<n[2]-sz[2]+stepsize[2])
          {   
              k=n[2]-sz[2];
              for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
      {
        for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
          
          /* add single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
          
        }
        if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
                for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
           }
      }
       if(j<n[1]-sz[1]+stepsize[1])
          {   
              j=n[1]-sz[1];
               for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
          
          /* add single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
          
        }
        if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
                for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              for (t=0; t<sz[0]; t++)
              {
                (x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += (b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0])[t];
              }
            }
          }
          blocknum++;
           }
          }
           }
    
    return;
}

