import numpy as np
import matplotlib.pyplot as plt

         #  ^y
Nx = 200 #  |
Ny = 50 #  | 
         #  (0,0)-->x

frames = 1000
rho0= 1.01
uxscale = 0.1
T= 0.6


f = np.zeros((Nx,Ny,9)) 


#rho * weighti * (1 + A + A^2/2 + B)
#A== (1/cs^2) * v @ u
#A^2 == (1/cs^4) * (v @ u)^2      #BOLTZMANN FOR EQUILIBRIUM
#B == -(u @ u) * (1/2*cs^2)

# 6  2  5
# 3  0  1
# 7  4  8

weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
f[...,:] = weights # initialising flow with desnity of 1 everywhere and no velocity
v = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)])
cs2 = 1/3 # sound speed squared
#v_shaped = np.zeros((Nx,Ny,9))
#v_shaped[...,:] = v
weights_shaped = np.zeros((Nx,Ny,9))
weights_shaped[...,:] = weights


n=5
def ux_inflow(y):
    ymax = (Ny-1)/2
    normalised = -1/((ymax+n)*(ymax-(Ny-1+n)))
    ux = uxscale * normalised * -(y+20)*(y-(Ny+19)) #f(y)
    if ux < 1e-4:
        raise ValueError("n needs to be bigger, ux too small for matrix inversion")
    return ux


#MATRIX FOR INFLOW
A = np.zeros((Ny,4,4))
C = np.zeros((Ny,4,9))
for y in range(Ny):
    a = 1/36 * (1 + ((1/cs2) * ux_inflow(y)) + (1/(2*cs2**2)) * ux_inflow(y)**2 -ux_inflow(y)**2 * (1/(2*cs2))) #rightward moving feqs
    b = 1/36 * (1 + ((1/cs2) * -ux_inflow(y)) + (1/(2*cs2**2)) * ux_inflow(y)**2 -ux_inflow(y)**2 * (1/(2*cs2))) #leftward
     #list of matricies for f(y) inflow conditions
    A[y] = np.array([              #x = ([rho], where Ax = B
        [ux_inflow(y),-1,-1,-1],   #    [f1],
        [1,-1,-1,-1],           #    [f5],
        [b-a,0,1,0],            #    [f8]
        [b-a,0,0,1]             #    )
        ])

    if abs(ux_inflow(y)) < 1e-8:
        raise ValueError("ux too small for Zou–He matrix inversion")

    #print("no error")
    if (np.linalg.det(A[y]) == 0) or (np.linalg.det(A[y]) < 0):
        print("error with matrix det")
        print(np.linalg.det(A[y]))
    #print(np.linalg.det(A))
    
    
    C[y] = np.array([
        [0,0,0,-1,0,0,-1,-1,0],
        [1,0,1,1,1,0,1,1,0],     #where this premultiplies vector f[0,:,:]
        [0,0,0,0,0,0,0,1,0],     #shape is 4x9 so we need to multiply by 9xn so transpose fslice
        [0,0,0,0,0,0,1,0,0]
        ])

M = np.linalg.solve(A,C)
#print(M.shape)


#debug
def cprint(*args, **kwargs):
    pass
    #print(*args, **kwargs)

#MATPLOTLIB
plt.ion()
fig, ax = plt.subplots()
modu = np.zeros((Nx,Ny))
img = ax.imshow(modu.T, origin='lower',cmap='Greys')
plt.colorbar(img)

#print(np.sum(f,axis=2).max())
#print("check1")


#for t in range(frames):
while True:

    #streaming step with boundary conditions considered
    f_streamed = np.zeros((Nx,Ny,9))

    f_streamed[:,:,0] = f[:,:,0] #velocity 0 doesnt get shifted
    f_streamed[1:,:,1] = f[:-1,:,1] #shifts right by 1 in x THIS LEAVES ALL THE f_streamed[0,:,1] FOR BOUNDARY CONDITIONS
    f_streamed[:,1:,2] = f[:,:-1,2] #shifts up by 1 so f[:,0,2] empty for boundaary (overlap at 0,0 and nx,ny etc can probably be wall conditions not inflow/outflow
    f_streamed[:-1,:,3] = f[1:,:,3] #shifts everything left, leaving all the far right points empty
    f_streamed[:,:-1,4] = f[:,1:,4] #shifts down
    f_streamed[1:,1:,5] = f[:-1,:-1,5] #shifts up and right - more boundary conditions
    f_streamed[:-1,1:,6] = f[1:,:-1,6] #shifts up and left
    f_streamed[:-1,:-1,7] = f[1:,1:,7]#shifts down and left
    f_streamed[1:,:-1,8] = f[:-1,1:,8] #shifts right and down



    #SOLID
    solid = np.zeros((Nx,Ny),dtype=bool)
    solid[25:35,20:30] = True
    for i,opp in [(1,3),(2,4),(3,1),(4,2),(5,7),(6,8),(7,5),(8,6)]:
        f_streamed[solid,i] = f_streamed[solid,opp]
        
    #BOUNDARY CONDITIONS SPECIFICALLY for i 1-8
    #inflow on left, outflow right and top bottom are reflective boundaries

    #REFLECTIVE BOUNDARIES ->  no slip so u = 0
    f_streamed[:,0,2] = f[:,0,4] #reflective boundary so the incoming velocity = outgoing e.g. 1 -> 3, 2<-->4- we dont think about oblique collisions as we are simply enforcing no slip
    f_streamed[:,(Ny-1),4] = f[:,(Ny-1),2]
    f_streamed[:,0,5] = f[:,0,7] 
    f_streamed[:,0,6] = f[:,0,8] 
    f_streamed[:,(Ny-1),7] = f[:,(Ny-1),5]
    f_streamed[:,(Ny-1),8] = f[:,(Ny-1),6]

    #print(np.sum(f_streamed,axis=2).max())
    cprint( "check2")
    
    #INFLOW/OUTFLOW
    #LEFT SIDE INFLOW- velocity constrained
    #MATRIX IS COMPUTED OUTSIDE LOOP
    #M has shape Ny,4,9
    f_slice = f_streamed[0,:,:] #shape is Ny,9
    #print(f_slice.shape)
    x = M @ f_slice[..., None]#None adds dimension so we have Ny,9,1 so numpy gets it. shape is Ny,4,1
    x = x.squeeze(-1) #makes shape Ny,4
    #print(x[:,0])
    f_streamed[0,:,1] = x[:,1]
    f_streamed[0,:,5] = x[:,2]
    f_streamed[0,:,8] = x[:,3]

    u = ((1/f_streamed.sum(axis=2)).T * np.einsum("ijk,kl->ijl",f_streamed,v).T).T
    cprint("u at inflow slice:", u[0,:])
    cprint("next")
    #print(np.sum(f,axis=2).max())
    #print( "check3") #BREAKS IN INFLOW
    
    #f_eq_in5 = rho0 * 1/36 * (1 + ((1/cs2) * ux_inflow) + (1/cs2**2) * ux_inflow**2 -ux_inflow**2 * (1/(2*cs2))) #i want to calculate inflow condition only once so outside loop
    #f_eq_in8 = rho0 * 1/36 * (1 + ((1/cs2) * ux_inflow) + (1/cs2**2) * ux_inflow**2 -ux_inflow**2 * (1/(2*cs2))) #u dot v on these diagonals with positive x is just ux
    #f_eq_in7 = rho0 * 1/36 * (1 + ((1/cs2) * -ux_inflow) + (1/cs2**2) * ux_inflow**2 -ux_inflow**2 * (1/(2*cs2))) 
    #f_eq_in6 = rho0 * 1/36 * (1 + ((1/cs2) * -ux_inflow) + (1/cs2**2) * ux_inflow**2 -ux_inflow**2 * (1/(2*cs2)))
    
    #f_streamed[0,:,5] = f_eq_in5 + (f_streamed[0,:,7] - f_eq_in7)
    #f_streamed[0,:,8] = f_eq_in8 + (f_streamed[0,:,6] - f_eq_in6)
    #a = f_streamed[0,:,3] + f_streamed[0,:,6] + f_streamed[0,:,7] - f_streamed[0,:,5] - f_streamed[0,:,8]
    #b = f_streamed[0,:,0] + f_streamed[0,:,2] + f_streamed[0,:,3] + f_streamed[0,:,4] + f_streamed[0,:,5] + f_streamed[0,:,6] + f_streamed[0,:,7] + f_streamed[0,:,8]
    #f_streamed[0,:,1] = (ux_inflow * b + a) / (1-ux_inflow)

    #RIGHT SIDE OUTFLOW- pressure constrained
    f_streamed[Nx-1, :, :] = 0.5*f_streamed[Nx-2, :, :] + 0.5*f_streamed[Nx-3,:,:]

    #ux_outflow = -1 + (f_streamed[(Nx-1),:,0] + f_streamed[(Nx-1),:,2] + f_streamed[(Nx-1),:,4] + 2*(f_streamed[(Nx-1),:,1] + f_streamed[(Nx-1),:,5] + f_streamed[(Nx-1),:,8]))/rho0
    #print(f_streamed[(Nx-1),:,:])
    #print(ux_outflow.max())
    #print(ux_outflow.min())
    #print("is this issue")
    #f_eq_out7 = rho0 * 1/36 * (1 + ((1/cs2) * -ux_outflow) + (1/(2*cs2**2)) * ux_outflow**2 -ux_outflow**2 * (1/(2*cs2))) #unsquared dot product is negative here
    #f_eq_out6 = rho0 * 1/36 * (1 + ((1/cs2) * -ux_outflow) + (1/(2*cs2**2)) * ux_outflow**2 -ux_outflow**2 * (1/(2*cs2)))
    #f_eq_out5 = rho0 * 1/36 * (1 + ((1/cs2) * ux_outflow) + (1/(2*cs2**2)) * ux_outflow**2 -ux_outflow**2 * (1/(2*cs2))) 
    #f_eq_out8 = rho0 * 1/36 * (1 + ((1/cs2) * ux_outflow) + (1/(2*cs2**2)) * ux_outflow**2 -ux_outflow**2 * (1/(2*cs2)))
    
    #f_streamed[(Nx-1),:,7] = f_eq_out7 + (f_streamed[(Nx-1),:,5] - f_eq_out5)
    #f_streamed[(Nx-1),:,6] = f_eq_out6 + (f_streamed[(Nx-1),:,8] - f_eq_out8)
    #c = f_streamed[(Nx-1),:,0] + f_streamed[(Nx-1),:,1] + f_streamed[(Nx-1),:,2] + f_streamed[(Nx-1),:,4] + f_streamed[(Nx-1),:,5] + f_streamed[(Nx-1),:,6] + f_streamed[(Nx-1),:,7] + f_streamed[(Nx-1),:,8]
    #f_streamed[(Nx-1),:,3] = c - rho0


    #CORNER BOUNDARY CONDITIONS NO SLIP + REFLECTION
    f_streamed[0,0,2] = f_streamed[0,0,4]
    f_streamed[0,0,1] = f_streamed[0,0,3]
    f_streamed[0,0,5] = f_streamed[0,0,7]
    

    f_streamed[(Nx-1),0,3] = f_streamed[(Nx-1),0,1]
    f_streamed[(Nx-1),0,2] = f_streamed[(Nx-1),0,4]
    f_streamed[(Nx-1),0,6] = f_streamed[(Nx-1),0,8]

    
    f_streamed[0,(Ny-1),4] = f_streamed[0,(Ny-1),2]
    f_streamed[0,(Ny-1),1] = f_streamed[0,(Ny-1),3]
    f_streamed[0,(Ny-1),8] = f_streamed[0,(Ny-1),6]

    f_streamed[(Nx-1),(Ny-1),3] = f_streamed[(Nx-1),(Ny-1),1]
    f_streamed[(Nx-1),(Ny-1),4] = f_streamed[(Nx-1),(Ny-1),2]
    f_streamed[(Nx-1),(Ny-1),7] = f_streamed[(Nx-1),(Ny-1),5]

    #CORNERS BCs
    #f_streamed[0,0,:] = f_streamed[1,0,:].sum() * weights
    #f_streamed[(Nx-1),0,:] = f_streamed[(Nx-2),0,:].sum() * weights #we take the rho from neighbour in x direction 
    #f_streamed[0,(Ny-1),:] = f_streamed[1,(Ny-1),:].sum() * weights
    #f_streamed[(Nx-1),(Ny-1),:] = f_streamed[(Nx-2),(Ny-1),:].sum() * weights


    f = f_streamed


    
    #collision_step
    f_eq = np.zeros((Nx,Ny,9)) #make rho which is size nx,ny,1, then find u which is the same, then do 9 f_eq slices, think matrix multipliation for dot product
    rho = f.sum(axis = 2) #summed over second axis (0,1,2) so now one value of rho at each point
    u = ((1/rho).T * np.einsum("ijk,kl->ijl",f,v).T).T #einstein summation over shared index (9 term stuff) maybe could be used?
    rhoxweights = (rho.T * weights_shaped.T).T
    f_eq = rhoxweights * (
        1
        + (1/cs2) * np.einsum("ijl,fl->ijf",u,v).T
        + (1/(2*cs2**2)) * (np.einsum("ijl,fl->ijf",u,v)).T**2
        - (1/(2*cs2)) * (np.sum(u*u,axis=2)).T
        ).T
    #print("test4")
    #print(f_eq[5,:])
    #print(f_eq[0,:])

    f += - (f-f_eq)/T

    #print("all of rho")
    #print((rho**2).min())
    A= (rho**2).min()
    cprint((rho**2).min())
    cprint(rho.min())
    #for i in range (Nx):
    #    for j in range (Ny):
    #        if rho[i,j]**2 == A and rho[i,j] != 1:
    #            print(i)
    #            print(j)
     #           print((rho**2).min())
            

    
    #print("is it a 0?")
    modu = np.sum(u*u,axis=2)**0.5
    img.set_data(modu.T)
    img.set_clim(0.0,modu.max()*1.1)
    #print(usquared.max())
    plt.pause(0.001)
    #print(
    #rho[50,50],
    #np.max(np.sqrt(np.sum(u*u, axis=2))),
    #np.min(rho),
    #np.max(rho)
    #)
 
    


plt.ioff()
plt.show()



