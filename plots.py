#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:13:51 2020

@author: ismini
"""

import re
from collections import defaultdict
import matplotlib.pyplot as mpl
import numpy as np


with open("run.output") as f:
    out1 = f.read()
    
with open("run_parallel.output") as f1:
    out = f1.read()
 
out1 = out1.strip().split('-')
    
seq = out1[1].split("\n")
seq= seq[1:]

nml_seq = [re.findall('n,m=(.+) l=(.+)', x) for x in seq[::3]]
t_seq = [re.findall('Time (.+)',x) for x in seq[2::3]]
t_seq = [float(x[0]) for x in t_seq]

nml_parallel = [re.findall('n,m=(.+) l=(.+) threads=(.+)', x) for x in parallel[::3]]
t_parallel = [re.findall('Time (.+)',x) for x in parallel[2::3]]
t_parallel = [float(x[0]) for x in t_parallel]
#l = [x[0] for x in l]

nml_parallel = [x[0] for x in nml_parallel]
d_parallel = defaultdict(list)
for x,y,z in nml_parallel:
    d_parallel[x].append((y,z))
    
len(d_parallel.keys())
threads = [x[1] for x in d_parallel['1'][0:13]]

#add 1 at the start of threads list
threads.insert(0,1)

    
out = out.strip().split("\n")
out[2+12]
out[13*12]
out[2*13*12]
t_sequential = [re.findall('Time (.+)',x) for x in seq[2::3]]
t_sequential = [float(x[0]) for x in t_sequential]

for i in range(0,10):
    
    #print(out[i*13*12])

    ###get execution time for n,m=1 and l=10
    
    #coarse_grained
    #out[0:13*12:12]
    cg = [re.findall('Time (.+)',x) for x in out[i*(13*12)+2:(i+1)*13*12:12]]
    #cg_1_10 = [re.findall('Time (.+)',x) for x in out[2:13*12:12]]
    cg = [float(x[0]) for x in cg]
    #cg_1_10 = [float(x[0]) for x in cg_1_10]
    
    #medium coarse-grained
    mcg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+5:(i+1)*13*12:12]]
    #mcg_1_10 = [re.findall('Time (.+)',x) for x in out[5:13*12:12]]
    mcg = [float(x[0]) for x in mcg]
    #mcg_1_10 = [float(x[0]) for x in mcg_1_10]
    
    #vectorized fine-grained
    vfg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+8:(i+1)*13*12:12]]
    #vfg_1_10 = [re.findall('Time (.+)',x) for x in out[8:13*12:12]]
    vfg = [float(x[0]) for x in vfg]
    #vfg_1_10 = [float(x[0]) for x in vfg_1_10]
    
    #fine-grained
    fg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+11:(i+1)*13*12:12]]
    #fg_1_10 = [re.findall('Time (.+)',x) for x in out[11:13*12:12]]
    fg = [float(x[0]) for x in fg]
    #fg_1_10 = [float(x[0]) for x in fg_1_10]
    
    #np arrays
    #np.append(A, [[7, 8, 9]], axis=0)
    Arr = np.array(cg, dtype=float_).reshape(int(len(cg)/13),13)
    #cg_1_10 = np.array(cg_1_10, dtype=float_).reshape(int(len(cg_1_10)/13),13)
    mcg = np.array(mcg, dtype=float_).reshape(int(len(mcg)/13),13)
    Arr = np.append(Arr, mcg, axis=0)
    #mcg_1_10 = np.array(mcg_1_10, dtype=float_).reshape(int(len(mcg_1_10)/13),13)
    vfg = np.array(vfg, dtype=float_).reshape(int(len(vfg)/13),13)
    Arr = np.append(Arr, vfg, axis=0)
    #vfg_1_10 = np.array(vfg_1_10, dtype=float_).reshape(int(len(vfg_1_10)/13),13)
    fg = np.array(fg, dtype=float_).reshape(int(len(fg)/13),13)
    Arr = np.append(Arr, fg, axis=0)
    #fg_1_10 = np.array(fg_1_10, dtype=float_).reshape(int(len(fg_1_10)/13),13)
    
    t_seq = np.full((1, 13), t_sequential[i])
    #t_seq_1_10 = np.full((1, 13), t_seq[0])
    #t_seq[0,0]
    
    #speedup values
    Arr_speedup = t_seq/cg
    #speed_up_cg_1_10 = t_seq_1_10/cg_1_10
    speed_up_mcg = t_seq/mcg
    Arr_speedup = np.append(Arr_speedup, speed_up_mcg, axis=0)
    #speed_up_mcg_1_10 = t_seq_1_10/mcg_1_10
    speed_up_vfg = t_seq/vfg
    Arr_speedup = np.append(Arr_speedup, speed_up_vfg, axis=0)
    #speed_up_vfg_1_10 = t_seq_1_10/vfg_1_10
    speed_up_fg = t_seq/fg
    Arr_speedup = np.append(Arr_speedup, speed_up_fg, axis=0)
    #speed_up_fg_1_10 = t_seq_1_10/fg_1_10
    #Arr_speedup.shape
    
    #append 0 at the first column of each speedup array
    Arr_speedup = np.insert(Arr_speedup, 0,np.zeros((1,4), dtype=float_), axis=1)
    #speed_up_cg = np.insert(speed_up_cg, 0 , 0)
    #speed_up_cg_1_10 = np.insert(speed_up_cg_1_10, 0 , 0)
    #speed_up_mcg = np.insert(speed_up_mcg, 0 , 0)
    #speed_up_mcg_1_10 = np.insert(speed_up_mcg_1_10, 0 , 0)
    #speed_up_vfg = np.insert(speed_up_vfg, 0 , 0)
    #speed_up_vfg_1_10 = np.insert(speed_up_vfg_1_10, 0 , 0)
    #speed_up_fg = np.insert(speed_up_fg, 0 , 0)
    #speed_up_fg_1_10 = np.insert(speed_up_fg_1_10, 0 , 0)
    
    #append seq time values at the first column of each numpy array
    Arr = np.insert(Arr, 0, np.full((1,4), t_seq[0,0]), axis=1)
    #cg = np.insert(cg, 0, t_seq[i])
    #cg_1_10 = np.insert(cg_1_10, 0, t_seq[0])
    #mcg = np.insert(mcg, 0, t_seq[i])
    #mcg_1_10 = np.insert(mcg_1_10, 0, t_seq[0])
    #vfg = np.insert(vfg, 0, t_seq[i])
    #vfg_1_10 = np.insert(vfg_1_10, 0, t_seq[0])
    #fg = np.insert(fg, 0, t_seq[i])
    #fg_1_10 = np.insert(fg_1_10, 0, t_seq[0])
    
    ###plots
    
    fig = mpl.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text('execution time')
    
    
    if i==0:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,10))
    elif i==1:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,100))
    elif i==2:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,1000))
    elif i==3:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,10))
    elif i==4:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,100))
    elif i==5:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,1000))
    elif i==6:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,10))
    elif i==7:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,100))
    elif i==8:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,1000))
    
       
    mpl.plot(threads, Arr[0,], label= "coarse_grained", c = 'orange')
    #mpl.plot(threads, cg_1_10, label= "coarse_grained", c = 'orange')
    mpl.plot(threads,Arr[1,], label= "medium coarse-grained", c= 'green')
    #mpl.plot(threads,mcg_1_10, label= "medium coarse-grained", c= 'green')
    mpl.plot(threads,Arr[2,], label= "vectorised fine-grained", c= 'red')
    #mpl.plot(threads,vfg_1_10, label= "vectorised fine-grained", c= 'red')
    mpl.plot(threads,Arr[3,], label= "fine-grained", c= 'blue')
    #mpl.plot(threads,fg_1_10, label= "fine-grained", c= 'blue')
    
    
    mpl.legend()
    ylabel('time')
    xlabel("# threads")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text('speedup')
    
    if i==0:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,10))
    elif i==1:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,100))
    elif i==2:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1,1000))
    elif i==3:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,10))
    elif i==4:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,100))
    elif i==5:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(100,1000))
    elif i==6:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,10))
    elif i==7:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,100))
    elif i==8:
        mpl.plot([], [], ' ', label="n,m={} & l={}".format(1000,1000))
    
    
    mpl.plot(threads, Arr_speedup[0,], label= "speedup coarse_grained", linestyle='dashed',c = 'orange')
    #mpl.plot(threads, speed_up_cg_1_10, label= "speedup coarse_grained", linestyle='dashed',c = 'orange')
    
    mpl.plot(threads, Arr_speedup[1,], label= "speedup medium coarse-grained", linestyle='dashed',c= 'green')
    #mpl.plot(threads, speed_up_mcg_1_10, label= "speedup medium coarse-grained", linestyle='dashed',c= 'green')
    mpl.plot(threads, Arr_speedup[2,], label= "speedup vectorised fine-grained", linestyle='dashed',c= 'red')
    #mpl.plot(threads, speed_up_vfg_1_10, label= "speedup vectorised fine-grained", linestyle='dashed',c= 'red')
    mpl.plot(threads, Arr_speedup[3,], label= "speedup fine-grained", linestyle='dashed',c= 'blue')
    #mpl.plot(threads, speed_up_fg_1_10, label= "speedup fine-grained", linestyle='dashed',c= 'blue')
    
    mpl.legend()
    ylabel('time seq / time parallel')
    xlabel("# threads")
    
    
    #mpl.show()
    fig.savefig('project'+str(i)+'.png', dpi=fig.dpi)
#################################################################

#n,m = 10000 l=10
    
out[10*13*12]
i=9
threads = threads[:-1]

cg = [re.findall('Time (.+)',x) for x in out[i*(13*12)+2:(i+1)*13*12:12]]
cg = [float(x[0]) for x in cg[:-1]]

#medium coarse-grained
mcg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+5:(i+1)*13*12:12]]
mcg = [float(x[0]) for x in mcg[:-1]]

#vectorized fine-grained
vfg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+8:(i+1)*13*12:12]]
vfg = [float(x[0]) for x in vfg[:-1]]

#fine-grained
fg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+11:(i+1)*13*12:12]]
fg = [float(x[0]) for x in fg[:-1]]

Arr = np.array(cg, dtype=float_).reshape(int(len(cg)/12),12)
mcg = np.array(mcg, dtype=float_).reshape(int(len(mcg)/12),12)
Arr = np.append(Arr, mcg, axis=0)
vfg = np.array(vfg, dtype=float_).reshape(int(len(vfg)/12),12)
Arr = np.append(Arr, vfg, axis=0)
fg = np.array(fg, dtype=float_).reshape(int(len(fg)/12),12)
Arr = np.append(Arr, fg, axis=0)

t_seq = np.full((1, 12), t_sequential[i])

#speedup values
Arr_speedup = t_seq/cg
speed_up_mcg = t_seq/mcg
Arr_speedup = np.append(Arr_speedup, speed_up_mcg, axis=0)
speed_up_vfg = t_seq/vfg
Arr_speedup = np.append(Arr_speedup, speed_up_vfg, axis=0)
speed_up_fg = t_seq/fg
Arr_speedup = np.append(Arr_speedup, speed_up_fg, axis=0)

#append 0 at the first column of each speedup array
Arr_speedup = np.insert(Arr_speedup, 0,np.zeros((1,4), dtype=float_), axis=1)
   
#append seq time values at the first column of each numpy array
Arr = np.insert(Arr, 0, np.full((1,4), t_seq[0,0]), axis=1)

fig = mpl.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.title.set_text('execution time')


if i==9:
    mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,10))
  
mpl.plot(threads, Arr[0,], label= "coarse_grained", c = 'orange')
mpl.plot(threads,Arr[1,], label= "medium coarse-grained", c= 'green')
mpl.plot(threads,Arr[2,], label= "vectorised fine-grained", c= 'red')
mpl.plot(threads,Arr[3,], label= "fine-grained", c= 'blue')


mpl.legend()
ylabel('time')
xlabel("# threads")

ax2 = fig.add_subplot(1, 2, 2)
ax2.title.set_text('speedup')

if i==9:
    mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,10))

mpl.plot(threads, Arr_speedup[0,], label= "speedup coarse_grained", linestyle='dashed',c = 'orange')
mpl.plot(threads, Arr_speedup[1,], label= "speedup medium coarse-grained", linestyle='dashed',c= 'green')
mpl.plot(threads, Arr_speedup[2,], label= "speedup vectorised fine-grained", linestyle='dashed',c= 'red')
mpl.plot(threads, Arr_speedup[3,], label= "speedup fine-grained", linestyle='dashed',c= 'blue')

mpl.legend()
ylabel('time seq / time parallel')
xlabel("# threads")


#mpl.show()
fig.savefig('project'+str(i)+'.png', dpi=fig.dpi)


########################################################
##n,m =10000 l=100 & l=1000

out[10*13*12]
out[10*13*12 + 9*12]
#the end:
out[10*13*12 + 9*12 + 9*12 -1]

i=10
threads = threads[:-3]

cg = [re.findall('Time (.+)',x) for x in out[i*(13*12)+2:i*(13*12)+9*12:12]]
cg = [float(x[0]) for x in cg]

#medium coarse-grained
mcg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+5:i*(13*12)+9*12:12]]
mcg = [float(x[0]) for x in mcg]

#vectorized fine-grained
vfg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+8:i*(13*12)+9*12:12]]
vfg = [float(x[0]) for x in vfg]

#fine-grained
fg =  [re.findall('Time (.+)',x) for x in out[i*(13*12)+11:i*(13*12)+9*12:12]]
fg = [float(x[0]) for x in fg]

Arr = np.array(cg, dtype=float_).reshape(int(len(cg)/9),9)
mcg = np.array(mcg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, mcg, axis=0)
vfg = np.array(vfg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, vfg, axis=0)
fg = np.array(fg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, fg, axis=0)

t_seq = np.full((1, 9), t_sequential[i])

#speedup values
Arr_speedup = t_seq/cg
speed_up_mcg = t_seq/mcg
Arr_speedup = np.append(Arr_speedup, speed_up_mcg, axis=0)
speed_up_vfg = t_seq/vfg
Arr_speedup = np.append(Arr_speedup, speed_up_vfg, axis=0)
speed_up_fg = t_seq/fg
Arr_speedup = np.append(Arr_speedup, speed_up_fg, axis=0)

#append 0 at the first column of each speedup array
Arr_speedup = np.insert(Arr_speedup, 0,np.zeros((1,4), dtype=float_), axis=1)
   
#append seq time values at the first column of each numpy array
Arr = np.insert(Arr, 0, np.full((1,4), t_seq[0,0]), axis=1)

fig = mpl.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.title.set_text('execution time')


if i==10:
    mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,100))
  
mpl.plot(threads, Arr[0,], label= "coarse_grained", c = 'orange')
mpl.plot(threads,Arr[1,], label= "medium coarse-grained", c= 'green')
mpl.plot(threads,Arr[2,], label= "vectorised fine-grained", c= 'red')
mpl.plot(threads,Arr[3,], label= "fine-grained", c= 'blue')


mpl.legend()
ylabel('time')
xlabel("# threads")

ax2 = fig.add_subplot(1, 2, 2)
ax2.title.set_text('speedup')

if i==10:
    mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,100))

mpl.plot(threads, Arr_speedup[0,], label= "speedup coarse_grained", linestyle='dashed',c = 'orange')
mpl.plot(threads, Arr_speedup[1,], label= "speedup medium coarse-grained", linestyle='dashed',c= 'green')
mpl.plot(threads, Arr_speedup[2,], label= "speedup vectorised fine-grained", linestyle='dashed',c= 'red')
mpl.plot(threads, Arr_speedup[3,], label= "speedup fine-grained", linestyle='dashed',c= 'blue')

mpl.legend()
ylabel('time seq / time parallel')
xlabel("# threads")


#mpl.show()
fig.savefig('project'+str(i)+'.png', dpi=fig.dpi)


#########################################################
##n,m = 10000 & l=1000

start = i*(13*12)+9*12
end = start + 9*12
cg = [re.findall('Time (.+)',x) for x in out[start+2:end:12]]
cg = [float(x[0]) for x in cg]

#medium coarse-grained
mcg =  [re.findall('Time (.+)',x) for x in out[start+5:end:12]]
mcg = [float(x[0]) for x in mcg]

#vectorized fine-grained
vfg =  [re.findall('Time (.+)',x) for x in out[start+8:end:12]]
vfg = [float(x[0]) for x in vfg]

#fine-grained
fg =  [re.findall('Time (.+)',x) for x in out[start+11:end:12]]
fg = [float(x[0]) for x in fg]

Arr = np.array(cg, dtype=float_).reshape(int(len(cg)/9),9)
mcg = np.array(mcg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, mcg, axis=0)
vfg = np.array(vfg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, vfg, axis=0)
fg = np.array(fg, dtype=float_).reshape(int(len(cg)/9),9)
Arr = np.append(Arr, fg, axis=0)

t_seq = np.full((1, 9), t_sequential[11])

#speedup values
Arr_speedup = t_seq/cg
speed_up_mcg = t_seq/mcg
Arr_speedup = np.append(Arr_speedup, speed_up_mcg, axis=0)
speed_up_vfg = t_seq/vfg
Arr_speedup = np.append(Arr_speedup, speed_up_vfg, axis=0)
speed_up_fg = t_seq/fg
Arr_speedup = np.append(Arr_speedup, speed_up_fg, axis=0)

#append 0 at the first column of each speedup array
Arr_speedup = np.insert(Arr_speedup, 0,np.zeros((1,4), dtype=float_), axis=1)
   
#append seq time values at the first column of each numpy array
Arr = np.insert(Arr, 0, np.full((1,4), t_seq[0,0]), axis=1)

fig = mpl.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.title.set_text('execution time')



mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,1000))
  
mpl.plot(threads, Arr[0,], label= "coarse_grained", c = 'orange')
mpl.plot(threads,Arr[1,], label= "medium coarse-grained", c= 'green')
mpl.plot(threads,Arr[2,], label= "vectorised fine-grained", c= 'red')
mpl.plot(threads,Arr[3,], label= "fine-grained", c= 'blue')


mpl.legend()
ylabel('time')
xlabel("# threads")

ax2 = fig.add_subplot(1, 2, 2)
ax2.title.set_text('speedup')


mpl.plot([], [], ' ', label="n,m={} & l={}".format(10000,1000))

mpl.plot(threads, Arr_speedup[0,], label= "speedup coarse_grained", linestyle='dashed',c = 'orange')
mpl.plot(threads, Arr_speedup[1,], label= "speedup medium coarse-grained", linestyle='dashed',c= 'green')
mpl.plot(threads, Arr_speedup[2,], label= "speedup vectorised fine-grained", linestyle='dashed',c= 'red')
mpl.plot(threads, Arr_speedup[3,], label= "speedup fine-grained", linestyle='dashed',c= 'blue')

mpl.legend()
ylabel('time seq / time parallel')
xlabel("# threads")


#mpl.show()
fig.savefig('project'+str(11)+'.png', dpi=fig.dpi)




###################################################################################
out1 = out1.strip().split('-')
#list(enumerate(out1))
parallel = out1[0].split("\n")
parallel = parallel[0:-1]

seq = out1[1].split("\n")
seq= seq[1:]

nml_seq = [re.findall('n,m=(.+) l=(.+)', x) for x in seq[::3]]
t_seq = [re.findall('Time (.+)',x) for x in seq[2::3]]
t_seq = [float(x[0]) for x in t_seq]

nml_parallel = [re.findall('n,m=(.+) l=(.+) threads=(.+)', x) for x in parallel[::3]]
t_parallel = [re.findall('Time (.+)',x) for x in parallel[2::3]]
t_parallel = [float(x[0]) for x in t_parallel]
#l = [x[0] for x in l]

nml_parallel = [x[0] for x in nml_parallel]
d_parallel = defaultdict(list)
for x,y,z in nml_parallel:
    d_parallel[x].append((y,z))
    
len(d_parallel.keys())
threads = [x[1] for x in d_parallel['1'][0:13]]

156/13

#each row holds times for each thread for the same l
t_array = np.array(t_parallel, dtype=float_).reshape(int(len(t_parallel)/13),13)
t_array.shape
#a = numpy.arange(5)
#for x in range(range(len(nml_parallel)),9):
    
    
    
time_n1 = t_parallel[0:13]

y_min = min(time_n1) -0.001
y_max = max(time_n1) +0.001
ylim(y_min, y_max)

#append seq time values at the first column of t_array
    
t_array = np.insert(t_array, 0, t_seq, axis=1)

#add 1 at the start of threads list
threads.insert(0,1)

#speed up values
speed_up = np.array(t_parallel, dtype=float_).reshape(int(len(t_parallel)/13),13)
speed_up.shape

t_seq_arr = np.array(t_seq, dtype=float_).reshape(12,1)
#t_seq_array = np.broadcast_to(t_seq_arr,(12,13))
sp_up_val = t_seq_arr/speed_up
#insert a column with 0 before the first column of sp_up_val for the seq code where speed up is equal to 0
sp_up_val = np.insert(sp_up_val, 0,np.zeros((12,), dtype=float_), axis=1)
    
for i in range(0,12,3):
    
    
    fig = mpl.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text('run time')
    
    
    if i==0:
        mpl.plot([], [], ' ', label="n,m={}".format(1))
    elif i==3:
        mpl.plot([], [], ' ', label="n,m={}".format(100))
    elif i == 6:
        mpl.plot([], [], ' ', label="n,m={}".format(1000))
    elif i == 9:
        mpl.plot([], [], ' ', label="n,m={}".format(10000))
   
    mpl.plot(threads, t_array[i,], label= "l=10", c = 'orange')
    mpl.plot(threads,t_array[i+1,], label= "l=100", c= 'green')
    mpl.plot(threads,t_array[i+2,], label= "l=1000", c= 'red')
    
    mpl.legend()
    ylabel('time')
    xlabel("# threads")

    #mpl.show()
    
    #mpl.figure()
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text('speedups')
    if i==0:
        mpl.plot([], [], ' ', label="n,m={}".format(1))
    elif i==3:
        mpl.plot([], [], ' ', label="n,m={}".format(100))
    elif i == 6:
        mpl.plot([], [], ' ', label="n,m={}".format(1000))
    elif i == 9:
        mpl.plot([], [], ' ', label="n,m={}".format(10000))
    mpl.plot(threads, sp_up_val[i,], label= "speedup (l=10)", linestyle='dashed',c = 'orange')
    mpl.plot(threads, sp_up_val[i+1,], label= "speedup (l=100)", linestyle='dashed',c= 'green')
    mpl.plot(threads, sp_up_val[i+2,], label= "speedup (l=1000)", linestyle='dashed',c= 'red')
    
    mpl.legend()
    ylabel('time seq / time parallel')
    xlabel("# threads")

   # mpl.show()
    fig.savefig('project'+str(i)+'.png', dpi=fig.dpi)


speed_up = np.array(t_parallel, dtype=float_).reshape(int(len(t_parallel)/13),13)
speed_up.shape
speed_up[10,]
t_seq_arr = np.array(t_seq, dtype=float_).reshape(12,1)
#t_seq_array = np.broadcast_to(t_seq_arr,(12,13))
sp_up_val = t_seq_arr/speed_up
#insert a column with 0 before the first column of sp_up_val for the seq code where speed up is equal to 0
sp_up_val = np.insert(sp_up_val, 0,np.zeros((12,), dtype=float_), axis=1)
#for x in range(12):
    #print(sp_up_val[x])
sp_up_val[0,]

np.zeros((12,1))
np.zeros((12,), dtype=float_)

#ax.set_xbound(lower = x_min-2, upper = x_max+2)
#ax.set_ybound(lower = y_min- y_min, upper = y_max + y_min)
#ax.get_xaxis().get_major_formatter().set_scientific(False)
  