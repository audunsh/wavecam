#Wavecam
#Finite difference wavesolver
#Audun Skau Hansen 2014

import numpy as np
from matplotlib.pyplot import *
import cv2

class fwave():
    def __init__(self, x,y,rho,b,q,I,V,f, dt):

        self.dt = dt
         
        self.dx = x[1]-x[0]
        self.x = x
        self.dy = y[1]-y[0]
        self.y = y
        self.rho = rho

        self.b = b
        self.beta =(1.0 + self.b*self.dt/(2.0*self.rho))**-1
        self.alpha = b*self.dt/(2.0*self.rho) - 1.0
        self.delta = (dt**2)/self.rho
        
        self.etax = (self.dt**2)/(2*self.rho*self.dx**2) 
        self.etay = (self.dt**2)/(2*self.rho*self.dy**2) 
        #self.alpha = rho/dt**2 - b/(2.0*dt)
        
        self.N = len(self.x)
        
        self.f = f
        self.u1 = I(self.x,self.y)
        self.RHS = 0*self.u1 #ensure same shape
        self.V = V(self.x,self.y)
        self.q = q(self.x,self.y)
        self.dt = (self.q.max() *self.dx)
        #print "Enforcing courant time step, dt:", self.dt 
        self.t = 0
        
    def initialize(self, t0):
        #set up domain and conditions
        self.u1 = self.I(self.x,self.y)
        self.RHS = self.u1
        self.t = t0
        
    def advance(self):
        #advance solution one time step
        self.u = self.beta * (2*self.u1 + self.rhs() + self.alpha*self.u2)
        
        #impose boundary conditions
        self.impose_boundaries()
        
        #update system
        self.update()
                
    def first_step(self):
        #perform modified first step
        #self.u = self.beta * (self.eta*self.rhs() + self.delta*self.u1 - self.alpha*(self.u1 - 2*self.dt*self.V))
        #impose boundary conditions
        self.u =.5* self.beta * (2.0*self.u1 + self.rhs() + self.alpha*(2.0*self.dt*self.V))
        self.impose_boundaries()
        
        
        #update system
        self.update()
        
    def update(self):
        self.t += self.dt
        self.u2 = self.u1
        self.u1 = self.u
        
    def rhs(self):


        q = self.q
        u1 = self.u1
        self.RHS *= 0

        self.RHS[1:-1,:] += (.5*(q[2:,:]+q[1:-1,:])*(u1[2:,:]-u1[1:-1,:]) - .5*(q[1:-1,:]+q[:-2,:])*(u1[1:-1,:]-u1[:-2,:]))*self.etax
        self.RHS[:,1:-1] += (.5*(q[:,2:]+q[:,1:-1])*(u1[:,2:]-u1[:,1:-1]) - .5*(q[:,1:-1]+q[:,:-2])*(u1[:,1:-1]-u1[:,:-2]))*self.etay

        return self.RHS + self.f(self.x, self.y, self.t)*self.delta        
        

    
    def solve(self, NT, display = False):
        self.first_step()
        for i in range(NT-1):
            self.advance()
            
            if display:
                cv2.imshow('frame',self.u)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    #Save picture to disk
                    a = time.strftime("%H:%M:%S")
                    print "Saved snapshot."
                    cv2.imwrite("wavesim" +a+".png" , image)
            
    def impose_boundaries(self):
        self.u[:,0 ] = self.u[:,1 ]
        self.u[:,-1] = self.u[:,-2 ]
        self.u[0,: ] = self.u[1,: ]
        self.u[-1,:] = self.u[-2,: ]
    
    def f(self, t):
        return 0


#Setting up domain    
def runtest():
    Lx = 10
    Ly = 10
    x = np.linspace(0,Lx,512)
    y = np.linspace(0,Ly,512)
    I = lambda x,y: 2*np.exp(-20*((x[:,None]-5)**2 + (y[:,None].T-5)**2)) 
    q = lambda x,y: np.ones((len(x), len(y))) #np.sin(x[:,None] + y[:,None].T)
    q = lambda x,y: 1 - .99*np.exp(-20*((x[:,None]-4)**2 + (y[:,None].T-3)**2)) 
    V = lambda x,y: np.zeros((len(x),len(y)))#0*x[:,None]*y[:,None].T 
    f = lambda x,y,t: 10*np.sin(3*t)*np.exp(-200*((x[:,None]-5)**2 + (y[:,None].T-5)**2)**2)
    eq = fwave(x,y,1,0,q,I,V,f,0.01)
    eq.solve(1000000, True)
    contourf(eq.u, cmap = "RdGy")
    colorbar()
    show()
    cv2.destroyAllWindows()
    
def test_constant_solution():
    Lx = 10
    Ly = 10
    x = np.linspace(0,Lx,512)
    y = np.linspace(0,Ly,512)
    C = 2.0
    I = lambda x,y: C *np.ones((len(x),len(y)))
    q = lambda x,y: np.ones((len(x), len(y))) #np.sin(x[:,None] + y[:,None].T)
    V = lambda x,y: np.zeros((len(x),len(y)))#0*x[:,None]*y[:,None].T 
    f = lambda x,y,t:  np.zeros((len(x),len(y)))
    
    eq = fwave(x,y,1,0,q,I,V,f,0.1)
    eq.solve(1000)
    if (abs(eq.u1.max() - C) + abs(eq.u1.min() - C)) == 0.0:
        print "Test succeeded for constant solution"

def stream_cam():

    #load webcam and iteratively solve the equation
    cap = cv2.VideoCapture(0)
    Nx = 240
    Ny = 320
    ret = cap.set(3,Ny)
    ret = cap.set(4,Nx)
    
    Lx = 48
    Ly = 64
    x = np.linspace(0,Lx,Nx)
    y = np.linspace(0,Ly,Ny)
    
    I = lambda x,y: 2*np.exp(-20*((x[:,None]-Lx*.5)**2 + (y[:,None].T-Ly*.5)**2)) 
    I = lambda x,y: np.zeros((len(x), len(y))) #np.sin(x[:,None] + y[:,None].T)
    q = lambda x,y: np.ones((len(x), len(y))) #np.sin(x[:,None] + y[:,None].T)
    q = lambda x,y: 1 - .99*np.exp(-20*((x[:,None]-4)**2 + (y[:,None].T-3)**2)) 
    V = lambda x,y: np.zeros((len(x),len(y)))#0*x[:,None]*y[:,None].T 
    f = lambda x,y,t: 10*np.sin(.5*t)*np.exp(-20*((x[:,None]-.5*Lx)**2 + (y[:,None].T-.5*Ly)**2)**2)
    
    eq = fwave(x,y,1,0,q,I,V,f,0.01)
    eq.first_step()
    while True:
        eq.advance()
        ret, frame = cap.read()
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/256.0
        #eq.q = im + .1
        eq.u1[im>.4] = 0 #dirichlet boundary
        eq.u2[im>.4] = 0 #dirichlet boundary
        im += eq.u
        cv2.imshow('frame',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
    cap.release()
    cv2.destroyAllWindows()

runtest()
#stream_cam()