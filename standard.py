import numpy as np
import math
import hdf5storage
from scipy.integrate import quad
from scipy.special import eval_hermite


def Step(x,a):
#Simplest possible definition of a step function
	return 0.5 *(np.sign(x-a) + 1.)
################################################################################

def locate(x,vec,WarnM=False):
################################################################################
#          Simple binary search of the value x in sorted vector vec.           #
#                                                                              #
#   If x is present in vec, will return exact indice                           #
#                                                                              #
#   If x not exactly present but in between two values in vec, will return the #
#   two indices that sandwich it                                               #
#                                                                              #
#   Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)   #
################################################################################

    jl = 0;
    ju = len(vec)-1

    if (x > vec[ju]) or (x==vec[ju]):
        if (x > vec[ju]) and (WarnM==True):
            print "Value x larger than any values in vec, returning last indice"
        return [int(ju)]
    elif (x<vec[jl]) or (x==vec[jl]):
        if (x<vec[jl]) and (WarnM==True):
            print "Value x smaller than any values in vec, returning first indice"
        return [int(jl)]
    else:
        req = 0
        while (ju-jl) > 1:
            jm = int(math.floor(0.5*(ju+jl)))
            if x == vec[jm]:
                jl = jm
                req = 1
                break
            elif x > vec[jm]:
                jl = jm
            else:
                ju = jm
        if req == 0:
            #return the indices that sandwich x in vec if not exactly there        
            indice = [int(jl),int(ju)]
        else:
            #return the exact position of x in vec if there 
            indice = [int(jl)]
        return indice
####################################################################################

def swap(A,i,j):
#swap the elements i and j in an array A 
	temp=A[i]
	A[i]=A[j]
	A[j]=temp
#################################################################################

def DataMatFormat(file_name):
#################################################################################
#		Open data saved in Matlab .mat format v7.3			#
#		The library hdf5storage must be installed			#
#	https://pythonhosted.org/hdf5storage/information.html#installation	#
#										#
# Return:									#
#										#
#  data    : a dictonary with keys the variables names and values the values of #
#	     the variables							#
#  InfoVar : a dictionary with keys the variables names and values the size of 	#
#	     variables (row,columns)						#
#################################################################################
	data=hdf5storage.loadmat(file_name)
	VarName=map(str,data.keys())
	SizeVar=map(lambda x: data[x].shape,data.keys())
	InfoVar=dict(zip(VarName,SizeVar))
	return data,InfoVar
#################################################################################



def  CompleteEllip(t,WithErr=False):
#"""
#Written by Louis-Francois Arsenault
#
#Calculate the complete elliptic integral of the first kind for argument
#not limited to real -1..1#
#
#"""	
	y=np.array(len(t))
	if WithErr is False:
		for r in range(0,len(t)):
			y[r] = quad(integrand_Ellip,0.,0.5*np.pi,args=(t[r]),epsabs=1e-10)[0]
		return y
	else: 	
		err=np.array(len(t))
		for r in range(0,len(t)):
                        Itemp = quad(integrand_Ellip,0.,0.5*np.pi,args=(t[r]),epsabs=1e-10)
			y[r]=temp[0]
			err[r]=temp[1]
		return y,err

def integrand_Ellip(theta,t):
	return 1./np.sqrt(1.-t**2*np.sin(theta)**2)
###################################################################################

def CheckVectors(v1,v2,tol):
#check if v1 and v2 are orthogonal or parralel
#return [ortho,par], where ortho can be True or False and par can be True or False

#tol acceptable tolerance to declare truth
	#will compare normalized vectors
	v11=v1.copy()/np.sqrt(v1.dot(v1))
	v22=v2.copy()/np.sqrt(v2.dot(v2))

	#Check ortho first
	if v11.dot(v22) <= tol:
		return [True,False]
	elif np.linalg.norm(np.cross(v11,v22)) <= tol:
		return [False,True]
	else:
		return [False,False]


def deriv_Gauss(x,n,sig):
#Derivative of the Gaussian function
#n order of the derivative
#ig standard deviation
#dG = Hermite poly time a Gaussian
	arg=x/(np.sqrt(2)*sig)
	G=1./(np.sqrt(2.*np.pi)*sig)*np.exp(-arg**2)
	if n==0:
		dG=G
	else:
		dG=(-1./(np.sqrt(2.)*sig))**n*eval_hermite(n,arg)*G
	return G,dG
