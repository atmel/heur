# !/usr/bin/env python

from subprocess import call

#RS

for pops in [1,10,100,1000]:
  for popSize in [32,64,128,256,512]:
    #print("===== pops=%d  popSize=%d  =====" %(pops, popSize))
    for i in range(0,50):
      call(['./RStest','%d' %(popSize),'%d' %(pops)])
      
#for pops in [1,10,100,1000]:
#  for popoff in [[32,64],[64,128],[128,256],[256,512],[512,1024]]:
    #print("===== pops=%d offspr=%d popSize=%d  =====" %(pops,popoff[1], popoff[0]))
#    for i in range(0,50):
#      call(['./GOtest','%d' %(popoff[0]),'%d' %(popoff[1]),'%d' %(pops)])