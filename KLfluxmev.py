#!usr/bin/python

# This code is designed to propagate dark matter particles through a depth of Earth's crust for a user-defined dark matter mass and cross section
# The code assumes an energy-independent total cross section, and a differential cross section that deviates from isotropic scattering only through the form factor
# The code samples from a path length distribution to determine the distance a particle travels into the crust before its first interaction
# At the depth of the first interaction, the code chooses the nucleus that the dark matter scatters with
# Based on the choice of nucleus, the code computes the range of possible energy transfer and uses the form factor to sample from the distribution of recoil energies and thus scattering angles
# The dark matter particle then loses energy and scatters at an angle determined by the above sampling
# If dark matter scatters into the atmosphere or drops below the energy required to trigger the detector, the code stops tracking it
# The energy required to trigger the detector is set to the lowest dark matter kinetic energy required to produce visible energy equal to the lowest-energy bin of the KL data used
# If a particle reaches detector depth, the code computes the proton recoil energy it would produce, and finally makes a histogram of recoil energies
# Additional comments throughout describe what is being done
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def integrand(x):
    birks=.015
    return (1/(1+birks*1.5/x**0.79))

def Te(energy):
    electronequiv=quad(integrand,0,energy)[0]
    return(electronequiv)

def espect(number,mchi,mp):
    spectrum=[]
    spect=[891.036, 600.703, 402.605, 268.504, 178.338, 118.019, 77.8463,51.183, 33.5467, 21.9229, 14.2847, 9.27905, 6.01075, 3.88292,2.50366, 1.61088, 1.03469, 0.663554, 0.42502, 0.271906, 0.173682,0.110747, 0.0704754, 0.0447336, 0.0283064, 0.0178411, 0.0111893,0.00697511, 0.00431557, 0.00264699, 0.00160619]
    en=[0.01, 0.0125893, 0.0158489, 0.0199526, 0.0251189, 0.0316228,0.0398107, 0.0501187, 0.0630957, 0.0794328, 0.1, 0.125893, 0.158489,0.199526, 0.251189, 0.316228, 0.398107, 0.501187, 0.630957, 0.794328,1., 1.25893, 1.58489, 1.99526, 2.51189, 3.16228, 3.98107, 5.01187,6.30957, 7.94328, 10.]
    func=interp1d(en,spect)
    while len(spectrum)<number:
        i=np.random.random_sample()
        j=np.random.random_sample()
        energy=.025+(1-.025)*i
        if func(energy)>j*func(.025):
            spectrum.append(energy)
            if len(spectrum) % 10000 == 0:
                print(len(spectrum))
    return(spectrum)

def mT(A):
    return (A * mp)

def mu(A,mchi):
    return( mT(A) * mchi / (mchi + mT(A)) )

def crosssec(A,sigma,mchi):
    return(sigma * A**2 * mu(A,mchi)**2/mu(1,mchi)**2)

def Fsqr(eR,A):
    return((1/(1+2*mp*eR/(.770*.770))**2)**2)

def F(eR, A):
    if A==1:
        return(1/(1+2*mp*eR/(.770*.770))**2)
    aF = 0.52
    sF = 0.9
    if eR>10**3:
        return(0)
    if eR<0:
        return (0)
    if eR == 0:
        return (1)
    else:
        qF = np.sqrt(2 * mT(A) * eR)
        cF = 1.23 * A ** (1. / 3.) - 0.6
        rF = np.sqrt(cF ** 2 + 7 * ((np.pi * aF) ** 2) / 3 - 5 * sF ** 2)
        qrF = qF * rF / 0.197
        return (3 * np.exp(-(qF * sF / 0.197) ** 2 / 2) * (np.sin(qrF) - np.cos(qrF) * qrF) / qrF ** 3) 

# Functions for sampling the scattering angle distribution
def randomnum(A,emax):
    while 0<1:
        i=np.random.random_sample()
        j=np.random.random_sample()
        if F(i*emax,A)**2>j*F(0,A)**2:
            break
    return i

def rCos(emax,A):
    return(2*(1-randomnum(A,emax))-1)

# probabilities to scatter with various nuclei
def chooseA(fSi,fOx,fAl,fFe):
    total=fSi+fOx+fAl+fFe
    random=np.random.random_sample()
    if random >= 0 and random < fSi/total:
        return 28
    if random >= fSi/total and random < (fSi + fOx)/total:
        return 16
    if random >= (fSi + fOx)/total and random < (fSi + fOx + fAl)/total:
        return 27
    if random >= (fSi + fOx + fAl)/total and random <= 1:
        return 56
    else:
        print("Danger")
        print(random)

# Flat phi distribution
def phi():
    return (2 * np.pi * np.random.random_sample())

# Useful constants
nAv=6.02214*10**23
rhocrust=2.7
mp=0.938

# KamLAND depth of 1000 meters, in cm
# User input: DM mass in GeV, cross section in cm^2/10^-30, number of particles to simulate
depth=100000
z=0
print(sys.argv[1])
mdm=float(sys.argv[1])
sigma=float(sys.argv[2])*10**-30
numpart=float(sys.argv[3])

# Nuclei in Earth's crust: mass number, abundance fraction, density, and finally the total mean free path
Si=28
Ox=16
Al=27
Fe=56

fSi=0.289
fOx=0.465
fAl=0.089
fFe=0.048

nSi=fSi*rhocrust*nAv/Si
nOx=fOx*rhocrust*nAv/Ox
nAl=fAl*rhocrust*nAv/Al
nFe=fFe*rhocrust*nAv/Fe

lambdainvSi=(nSi * crosssec(Si,sigma,mdm))
lambdainvOx=(nOx * crosssec(Ox,sigma,mdm))
lambdainvAl=(nAl * crosssec(Al,sigma,mdm))
lambdainvFe=(nFe * crosssec(Fe,sigma,mdm))
lambdatotal=1/(lambdainvSi+lambdainvOx+lambdainvAl+lambdainvFe)

# Generate a spectrum running from 100 eV to 1 GeV
bins1=np.logspace(np.log10(.0001),np.log10(1000),100)
espect=espect(numpart,mdm,mp)
normalizedcounts1=[]
xax1=[]
counts1,bins1,bars1=plt.hist(espect,bins=bins1,alpha=0.5)
plt.close()
for i in range(len(counts1)):
    countspermev1=counts1[i]/(bins1[i+1]-bins1[i])
    normalizedcounts1.append(countspermev1)
    xax1.append(bins1[i]*(bins1[i+1]/bins1[i])**(1/2))

# Threshold recoil energy and minimum DM energy to produce this recoil
eth=.0016
Emin = eth
if eth > 2.*mdm:
    Emin = (eth / 2. - mdm) * (1. + (1. + (2. * eth * (mdm + 1. * mp)**2.) / ((1.*mp) * (2. * mdm - eth)**2.) )**(1./2.))
else:
    Emin = (eth / 2. - mdm) * (1. - (1. + (2. * eth * (mdm + 1. * mp)**2.) / ((1.*mp) * (2. * mdm - eth)**2.) )**(1./2.))
print(Emin)

finalspect=[]
recoilspect=[]
equivspect=[]
reflect=[]
eloss=[]
angle=[]
i=0
# Loop over DM particles
for en in espect:
    z=0
    coscumulative=1
    energy=en
    ncol=0 # keep track of total number of collisions, just for my reference
    # Loop over scatterings
    while z < depth:
        distance = -lambdatotal*np.log(np.random.random_sample())
        # Particle travels vertical distance before scattering the first time, then after each scattering it travels in some direction depending on the sampled angle
        # These next three lines can also be put at the end of the loop, causing particle to scatter as soon as it reaches the crust
        z += distance * coscumulative
        if z > depth:
            break
        if z < 0:
            reflect.append(z)
            break
        A = chooseA(fSi,fOx,fAl,fFe)
        emax = (energy*energy + 2*mdm*energy)/(energy + (mdm + mT(A))*(mdm + mT(A))/(2*mT(A)))
        costhetacm = rCos(emax,A)

        # Once CM scattering angle is chosen, we need to compute the lab scattering angle (requiring relativistic kinematics)
        v_ini = np.sqrt(1 - 1 / (energy/mdm + 1)**2 )
        gamma = (energy + mdm + mT(A)) / np.sqrt(mdm**2 + mT(A)**2 + 2 * mT(A) * (energy + mdm))
        beta = np.sqrt((energy + mdm)**2 - mdm**2)/(energy + mdm + mT(A))
        betaprime = (v_ini - beta)/(1 - beta*v_ini)
        costhetalab = gamma * (beta / betaprime + costhetacm) / np.sqrt(1 + gamma**2 * (costhetacm + beta/betaprime)**2 - costhetacm**2)

        # Compute energy and angle wrt vertical after scattering
        energy = energy - emax*(1-costhetacm)/2
        ncol+=1
#        print(energy)
        coscumulative = coscumulative*costhetalab - np.sqrt(1-coscumulative**2)*np.sqrt(1-costhetalab**2)*np.cos(phi())

        # This is the alternate location for the three lines at the top
        # If it is not commented out, dark matter scatters as soon as it hits the top of the crust
##        z += distance * coscumulative
##        if z > depth:
##            break
        
        if energy < Emin:
            break
    eRec = (energy*energy + 2*mdm*energy)/(energy + (mdm + mT(1))*(mdm + mT(1))/(2*mT(1))) * (1 - rCos((energy**2+2*mdm*energy)/(energy+(mdm+1*mp)**2/(2*1*mp)),1)) / 2

    # If particle reaches detector with enough energy to possibly trigger detector, append recoil energy it would induce to array
    if z > 0 and energy > Emin:
        threshold,th2 = quad(Fsqr,0,(energy*energy + 2*mdm*energy)/(energy + (mdm + mT(1))*(mdm + mT(1))/(2*mT(1))),args=(1,))
        if threshold/((energy*energy + 2*mdm*energy)/(energy + (mdm + mT(1))*(mdm + mT(1))/(2*mT(1)))) > np.random.random_sample():
            finalspect.append(energy)
            recoilspect.append(eRec)
            equivspect.append(Te(eRec))
        print(ncol)
    # If it reaches detector with too little energy, append energy
    if z > 0 and energy < Emin:
        eloss.append(energy)
#        recoilspect.append(eRec)
#        equivspect.append(Te(eRec))
    if i % 10000==0:
        print("Number of particles is ",i)
    i+=1

print("Particles reaching detector =",len(finalspect))
print("Particles deflected into the atmosphere =",len(reflect))
print("Particles that lost too much energy =",len(eloss))

# Print final recoil spectrum, just for reference
print(finalspect)
print(recoilspect)
print(equivspect)
energyspect=finalspect+eloss

# Histogram of final recoil spectrum
bins=np.linspace(.005,.02,15)
counts,bins,bars=plt.hist(equivspect,bins=bins,alpha=0.5)
plt.close()
normalizedcounts=[]
xax=[]
for i in range(len(counts)):
    countspermev=counts[i]/(bins[i+1]-bins[i])/1000.
    testflux=.0000004086## flux in /cm^2/s for 1 MeV, 10^-30 cm^2 in the given energy range
    density=0.780*nAv/7.
    volume=600*300*300*3.14159
    exposure=123000000000./365*nAv/7.*3.15*10**7.
    normalized=countspermev/numpart*testflux*(sigma/10.**(-30.))*sigma*exposure*3./2.*3.1415*(bins[1]-bins[0])*1000
    normalizedcounts.append(normalized)
    xax.append(bins[i]*(bins[i+1]/bins[i])**(1./2.))
# the three lists printed are energy in GeV, dN/dE (number of events per MeV at those energies), and E*dN/dE
print(xax)
print(normalizedcounts)
ednde=[]
for i in range(len(xax)):
    ednde.append(xax[i]*normalizedcounts[i]*1000.)
print(ednde)
file=open("spectrummevklplot.txt","a")
file.write(str(normalizedcounts)+"\n")
file.close()
