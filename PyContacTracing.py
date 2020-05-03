# Covid-19 disease spreading multi-agents simulation
# Author: L. Gabrielli, 2020
# Due to computational limitations, the population of agents can't be excessively high. Therefore I'm modelling only a
# portion of the actual population. Therefore, I'm explicitly not modeling the saturation (herd immunity due widespread
# exposure to the virus), by enforcing that each person in incubation will always infect R0 people. This, however, will
# slow down simulation. I could speed up by using separate lists for each status. TODO.
# For simplicity, the recovered people are also assumed to be immune (as of May 3rd this has not been assessed yet, see
# https://www.who.int/news-room/commentaries/detail/immunity-passports-in-the-context-of-covid-19).
#
# This software is provided AS IS with no responsibility whatsoever on its usage, interpretation and the validity of the results.
#

from enum import Enum
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import math
import sys

### SIMULATION-RELATED PARAMETERS
RERUN_THE_EXPERIMENTS = 0  # make it 1 the first time you run the script
POP = 50e3  # population
DAYS = 120  # simulation duration
EFFICIENCY_REDUCTION = 0.1  # a portion of those who receive the alarm fail to go quarantine
DEFER_SWAB_DAYS = 2  # delay from symptoms to alarm because of long swab RTPCR processing time
DAY0_INFECTED = 500  # how many people are infected on day 0

# return codes
DO_INFECT = 1
DO_ALARM = 2


class status(Enum):
    healthy = 0
    incub = 1
    illquarant = 2
    recovered = 3
    dead = -1

### You can try change these
class humanParams:
    R0 = 3.0
    Ti = 7  # incubation time
    Tr = 21  # time to recovery
    Td = 15  # time to death
    Tq = Ti + 3  # quarantine: lasts more than known incubation for extra safety
    Pd = 0.2  # Death probability for ill people (in reality it should be lower)
    Pi = R0/Ti  # daily probability of infecting someone
    meeting = 30  # daily contacts, these are the people that come close enough to be infected and the App will record these
    amtHasApp = 0.0  # percent people having the App
    swabAvailabilityProbability = 1  # how many people will get a swab when they show symptoms. The others will just go quarantine

    def setR0(self,r0):
        self.R0 = r0
        self.Pi = r0/self.Ti

class human:
    st = status.healthy
    hp = []
    id = 0
    icounter = 0 # illness day counter
    riskcounter = 0
    willdie = False
    safequarantinectr = 0
    hasApp = 0
    meetingList = []
    defer_alarm = 0

    def __init__(self,id,humanParams):
        self.hp = humanParams
        self.id = id
        self.hasApp = random.random() < self.hp.amtHasApp
        self.meetingList = []

    def getId(self):
        return self.id

    def infect(self):
        if self.st == status.healthy:
            self.st = status.incub
            self.icounter = self.hp.Ti
            return 1
        else:
            return 0 # this guy is not infectable, try with a healthy one

    def incubend(self):
        self.willdie = random.random() < self.hp.Pd # decide if will die or not
        if self.willdie:
            self.icounter = self.hp.Td
        else:
            self.icounter = self.hp.Tr
        self.st = status.illquarant
        # if has app, now the guy will alert the others
        if self.hasApp:
            if random.random() < self.hp.swabAvailabilityProbability:
                if DEFER_SWAB_DAYS:
                    self.defer_alarm = DEFER_SWAB_DAYS
                else:
                    return DO_ALARM
        return 0

    def appAlarm(self,parent):
        if self.hasApp:
            # not all people will follow the App indication
            if random.random() > EFFICIENCY_REDUCTION:
                self.safequarantinectr = self.hp.Tq

    def quarantineEnd(self):
        self.st = status.healthy

    def isMeetable(self):
        if self.st == status.healthy and self.safequarantinectr == 0:
            return True
        else:
            return False

    def addMet(self,guyID):
        self.meetingList.append(guyID)

    def isDead(self):
        if self.st == status.dead:
            return 1
        else:
            return 0

    # knows to be infected
    def knowsIsInfected(self):
        if self.st == status.illquarant:
            return 1
        else:
            return 0

    # all infected either knows or not
    def isInfected(self):
        if self.st == status.illquarant or self.st == status.incub:
            return 1
        else:
            return 0

    def isRecovered(self):
        if self.st == status.recovered:
            return 1
        else:
            return 0

    def isHealthy(self):
        if self.st == status.healthy:
            return 1
        else:
            return 0

    def getStatus(self):
        return self.st

    # Step the simulation 1 day forward
    def step(self):
        retval = 0
        # if quarantined (either incubating or not)
        if self.safequarantinectr > 0:
            self.safequarantinectr -= 1
            if self.safequarantinectr == 0:
                self.quarantineEnd()
        # all other states
        if self.st == status.incub:
            if self.safequarantinectr < 1:  # if not in quarantine will infect
                self.riskcounter += self.hp.Pi  # increment, this will work only if Pi < 1 per day (reasonable)
                if self.riskcounter > 1:
                    self.riskcounter -= 1
                    retval += DO_INFECT
            self.icounter -= 1
            if self.icounter == 0:
                retval += self.incubend()
        if self.st == status.illquarant:
            self.icounter -= 1
            self.defer_alarm -= 1
            if self.defer_alarm == 0:
                retval = DO_ALARM
            if self.icounter == 0:
                if self.willdie:
                    self.st = status.dead
                else:
                    self.st = status.recovered
        return retval


class experiment():
    HP = []
    listpop = []

    def __init__(self,hp):
        self.HP = hp
        self.listpop = []
        for i in range(int(POP)):
            self.listpop.append( human(i,self.HP) )


        # start the simulation with a number of infected people
        for i in range(DAY0_INFECTED):
            self.listpop[i].infect()

    def run(self):
        # ITERATE
        print("###\t\t###\t\tINFEC\t\tDEATHS")
        ninfec = np.zeros(DAYS)
        ndead = np.zeros(DAYS)
        nrecovered = np.zeros(DAYS)
        ntotal = np.zeros(DAYS)
        nhealthy = np.zeros(DAYS)
        skipContagi = 0
        for day in range(DAYS):
            for i in range(int(POP)):
                retval = self.listpop[i].step()
                if retval & DO_INFECT:
                    OK = 0
                    deadlockctr = 0
                    while not OK and deadlockctr < int(POP) and not skipContagi:
                        randomguy = math.floor(random.random() * POP)
                        OK = self.listpop[randomguy].infect() # the rationale for using OK is just to avoid infecting someone who is not healthy
                        deadlockctr += 1
                if self.listpop[i].getStatus() == status.incub:
                    tomeet = self.HP.meeting
                    deadlockctr = 0
                    while tomeet > 0 and deadlockctr < 0.5*int(POP) and not skipContagi:
                        randomguy = math.floor(random.random() * POP)
                        if self.listpop[randomguy].isMeetable():
                            self.listpop[i].addMet(self.listpop[randomguy].getId())
                            tomeet -= 1
                        deadlockctr += 1
                if retval & DO_ALARM:
                    for j in self.listpop[i].meetingList:
                        self.listpop[j].appAlarm(self.listpop[i].getId())

                ninfec[day] += self.listpop[i].isInfected()
                ndead[day] += self.listpop[i].isDead()
                nrecovered[day] += self.listpop[i].isRecovered()
                nhealthy[day] += self.listpop[i].isHealthy()

            ntotal[day] = ninfec[day] + ndead[day] + nrecovered[day] + nhealthy[day]
            print("DAY\t\t#"+str(day)+":\t\t"+str(ninfec[day])+"\t\t"+str(ndead[day]))
            if nhealthy[day] < 0.3 * POP:
                skipContagi = 1  # trick to avoid slowing down simulation when it makes no sense
                print("AVOID POP STARVING AND CPU SLOWDOWN")
        return ninfec, ndead, nrecovered

#### MAIN ####
if RERUN_THE_EXPERIMENTS == 1:

    hp1 = humanParams()
    hp1.amtHasApp = 0.7
    hp1.swabAvailabilityProbability = 0.75
    experiment1 = experiment(hp1)
    ni1, nd1, nr1 = experiment1.run()

    NI1 = np.asarray(ni1)
    ND1 = np.asarray(nd1)
    NR1 = np.asarray(nr1)

    np.savez('outcomes3.npz',NI1=NI1, ND1=ND1, NR1=NR1)


else:

    npzfile = np.load('outcomes3.npz')
    ni1 = npzfile['NI1']
    nd1 = npzfile['ND1']
    nr1 = npzfile['NR1']




plt.plot(ni1, label='cases')
plt.plot(nd1, label='deceased')
plt.plot(nr1, label='recovered')
plt.legend()
plt.xlabel('days')
plt.ylabel('cases')
plt.title('A possible scenario')
plt.ylim((0, 7000))
plt.show()
