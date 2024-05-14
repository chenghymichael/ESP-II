# Heading: 0

import time
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

np.set_printoptions(precision=5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 4

########

r = 0.2
w = r/np.tan(30*np.pi/180)
h = w/np.tan(30*np.pi/180)

sampf = 10000
sampt = 1/sampf
rest = 1
allt = 60

t_dot = np.arange(0, allt, sampt)
t_sec = np.arange(0, allt, 1)
S_pos = np.array([[0, h], [w, 0], [-w, 0]]) 
centre = np.array([[0], [r]])

# Heading: 1. Generate Source Value

class Source:
    def __init__(self, pos, m, rawf):
        assert np.shape(pos)[0] == allt, "init source, pos length"
        assert np.shape(pos)[1] == 2, "init source, pos component"
        assert np.shape(m)[0] == allt, "init source, magnitude length"
        assert np.shape(rawf)[0] == allt, "init source, rawf length"

        self.pos = pos
        self.m = m
        self.rawf = rawf
        self.value = np.zeros(allt*sampf, dtype=float)

        self.rawt = 1/(self.rawf/60)
        self.qt = np.rint((self.rawt/sampt))
        self.qf = (1/(self.qt*sampt)) * 60

        tcount = sampf*2
        for i in range(0,allt*sampf):
            if (tcount >= self.qt[(int)(i/sampf)]):
                self.value[i] = self.m[(int)(i/sampf)]
                tcount = 1
            else:
                tcount += 1

# Heading: 2. Theoretical sensor values

class SensorM:
     def __init__(self, valuem):
        assert np.shape(valuem)[0] == allt*sampf, "init sensorM, valuem length"
        self.value = valuem

class Sensor:
    def __init__(self, posx, posy, valuem, pos1, value1, pos2 = np.zeros((allt*sampf, 2)), value2 = np.zeros(allt*sampf)):
        assert type(posx) == np.float64, "init sensor, posx type"
        assert type(posy) == np.float64, "init sensor, posy type"
        
        assert np.shape(valuem)[0] == allt*sampf, "init sensor, valuem length"
        
        assert np.shape(pos1)[0] == allt, "init sensor, pos1 length"
        assert np.shape(pos1)[1] == 2, "init sensor, pos1 component"
        assert np.shape(value1)[0] == allt*sampf, "init sensor, value1 length"

        assert np.shape(pos2)[0] == allt, "init sensor, pos2 length"
        assert np.shape(pos2)[1] == 2, "init sensor, pos2 component"
        assert np.shape(value2)[0] == allt*sampf, "init sensor, value2 length"
        
        self.posx = posx
        self.posy = posy
        self.value = np.zeros(allt*sampf, dtype=float)
        self.freq_domain = []
        
        for i in range(0,allt*sampf):
            index = (int)(i/sampf)
            self.value[i] = (value1[i] / ( (pos1[index,0]-self.posx)**2 + (pos1[index,1]-self.posy)**2 )) + valuem[i]
            self.value[i] += value2[i] / ( (pos2[index,0]-self.posx)**2 + (pos2[index,1]-self.posy)**2 )
    
    def read_freq(self):
        return self.freq_domain

# Heading: 3. Frequency domain for each sensor

def search_np(array, value, buffer, mode):
    if mode == 1:
        count = 0
        for x in array:
            if abs(x - value) < buffer:
                return count
            count += 1
        return -1
    elif mode == 2:
        count = 0
        for x in array:
            if abs(x - value)/value < buffer:
                return count
            count += 1
        return -1
    else:
        return -1

def get_freq_domain(value):
    assert np.shape(value)[0] == allt*sampf, "get_freq_domain, value length"
    buffer1 = 0.00001
    buffer2 = 0.05

    tot_freq_domain = []
    for i in range(0,allt):
        freq_domain = np.empty((0,4), dtype=float)
        for j in range (i*sampf, (i+1)*sampf):
            if value[j] != 0:
                mag = value[j]
                index = search_np(freq_domain[:,0], mag, buffer1, 1)
                if (index == -1):
                    freq_domain = np.append(freq_domain, np.array([[mag,j,0,0]]), axis = 0)
                else: 
                    if freq_domain[index, 2] == 0:
                        freq_domain[index, 2] = j
        
        freq_domain[:,3] = freq_domain[:,2] - freq_domain[:,1]
        
        # cant find f within 1s
        if (np.shape(np.where(freq_domain[:,3] <= 0))[1] > 0):
            if i < allt-1:
                # look into next second
                for j in range ((i+1)*sampf, (i+2)*sampf):
                    if value[j] != 0:
                        mag = value[j]
                        index = search_np(freq_domain[:,0], mag, buffer2, 2)
                        if (index >= 0):
                            if freq_domain[index, 2] <= 0:
                                freq_domain[index, 2] = j
                
                freq_domain[:,3] = freq_domain[:,2] - freq_domain[:,1]

            else: 
                # last second
                for j in range(0, np.shape(freq_domain)[0]):
                    if freq_domain[j,3] <= 0:
                        freq_domain[j,3] = sampf
        
        freq_domain = np.delete(freq_domain, freq_domain[:,3] <= 0, 0)
        freq_domain = np.delete(freq_domain, freq_domain[:,3] > sampf, 0)

        freq_domain = freq_domain[np.argsort(freq_domain[:,3])]
        freq_domain = np.delete(freq_domain, [1, 2], 1)
        tot_freq_domain.append(freq_domain)

    return tot_freq_domain

# Heading: 4. Triangulation & Output

def is_under_coverage(x, y):
    if (x >= w) or (x <= -w):
        return False
    if (y >= h) or (y <= 0):
        return False
    
    if x >= 0:
        return y <= ((-1) * h / w) * x + h
    else:
        return y <= (h / w) * x + h

def check_result(x, y, M, Sam, Sbm, Scm, S_pos):
    Sar = M / ((S_pos[0,0] - x)**2 + (S_pos[0,1] - y)**2)
    Sbr = M / ((S_pos[1,0] - x)**2 + (S_pos[1,1] - y)**2)
    Scr = M / ((S_pos[2,0] - x)**2 + (S_pos[2,1] - y)**2)

    Sad = abs(Sar - Sam)
    Sbd = abs(Sbr - Sbm)
    Scd = abs(Scr - Scm)

    if max(Sad, Sbd, Scd) > 0.00001:
        return False
    else:
        return True

def eqs(x, Sa, Sb, Sc):
    eq = []
    eq.append(x[2] / ((x[0] - S_pos[0,0])**2 + (x[1] - S_pos[0,1])**2) - Sa)
    eq.append(x[2] / ((x[0] - S_pos[1,0])**2 + (x[1] - S_pos[1,1])**2) - Sb)
    eq.append(x[2] / ((x[0] - S_pos[2,0])**2 + (x[1] - S_pos[2,1])**2) - Sc)
    return eq

def triangulation(S1a, S1b, S1c):
    assert (type(S1a) == np.float64) or (type(S1a) == int), "triangulation, S1a type"
    assert (type(S1b) == np.float64) or (type(S1b) == int), "triangulation, S1b type"
    assert (type(S1c) == np.float64) or (type(S1c) == int), "triangulation, S1c type"

    ans = np.ones(3, dtype=float)*-1
    yguess = 0

    while (((is_under_coverage(ans[0], ans[1]) == False) or (check_result(ans[0], ans[1], ans[2], S1a, S1b, S1c, S_pos) == False)) and (yguess < h)):
        ans = so.fsolve(eqs, [0,yguess,1], args=(S1a, S1b, S1c))
        yguess += 0.01

    if (is_under_coverage(ans[0], ans[1]) == False) or (check_result(ans[0], ans[1], ans[2], S1a, S1b, S1c, S_pos) == False):
        return np.array([0,0,1])
    
    return ans

########

class Results:
    def __init__(self, Sain, Sbin, Scin, S1in, S2in):
        
        self.Sa = Sain.freq_domain
        self.Sb = Sbin.freq_domain
        self.Sc = Scin.freq_domain
        self.left_qt = np.ones(allt, dtype=float) * sampf
        self.right_qt = np.ones(allt, dtype=float) * sampf

        self.left_bpm = np.ones(allt, dtype=float)
        self.right_bpm = np.ones(allt, dtype=float)

        self.S1 = S1in
        self.S2 = S2in

        self.num = fetus_num

        self.error = 0.0
        self.accuracy = 0.0
        self.tot_count = self.num * allt
        
    def compute(self):
        for i in range(0, allt):
            num = np.shape(self.Sa[i])[0]
            for j in range(0,num):
                qt = self.Sa[i][j,1]

                S1a = self.Sa[i][j,0]
                
                indexb = search_np(self.Sb[i][:,1], qt, 0.00001, 1)
                if indexb == -1:
                    S1b = 1
                else:
                    S1b = self.Sb[i][indexb,0]
                
                indexc = search_np(self.Sc[i][:,1], qt, 0.00001, 1)
                if indexc == -1:
                    S1c = 1
                else:
                    S1c = self.Sc[i][indexc,0]
                
                source = triangulation(S1a, S1b, S1c)

                if num == 1:
                    self.left_qt[i] = qt
                    self.right_qt[i] = qt
                else:
                    if source[0] >= 0:
                        self.right_qt[i] = qt
                    else:
                        self.left_qt[i] = qt

    def to_bpm(self):
        self.left_bpm = 1 / (self.left_qt/sampf) * 60
        self.right_bpm = 1 / (self.right_qt/sampf) * 60

    def get_accuracy(self):
        if self.num == 1:
            for i in range(0, allt):
                e = abs(self.left_bpm[i] - self.S1.qf[i])
                self.error += e
                if e/self.S1.qf[i] < 0.1:
                    self.accuracy += 1
        else:
            for i in range(0, allt):
                e1 = abs(self.left_bpm[i] - self.S1.qf[i])
                e2 = abs(self.right_bpm[i] - self.S2.qf[i])
                self.error += e1 + e2
                if e1/self.S1.qf[i] < 0.1:
                    self.accuracy += 1
                if e2/self.S2.qf[i] < 0.1:
                    self.accuracy += 1

# Heading: 5. Generate Random

def rand(low, high, mode):
    # mode 0: int, mode 1: float
    range = high - low
    num = random.random() * range + low
    if mode == 0:
        num = round(num)
    return num

def rand_pos(mode):
    # mode 0: all, mode 1: left, mode 2: right
    assert ((0<=mode<=2) == True), "rand_pos mode"

    if mode == 0:
        low = (-w) * 0.9
        high = w * 0.9
    elif mode == 1:
        low = (-w) * 0.9
        high = -0.03
    else:
        low = 0.03
        high = w * 0.9
    
    x = rand(low, high, 1)

    if x >= 0:
        y = (((-1) * h / w) * x + h) * rand(0.05, 0.9, 1)
    else:
        y = ((h / w) * x + h) * rand(0.05, 0.9, 1)

    return [x, y]

def rand_moving(low, high, maxd, length, mode):
    # mode 0: int, mode 1: float

    list = np.empty(length, dtype=float)
    num = rand(low, high, 1)

    for i in range(0, length):
        if (i % 3 == 0):
            delta = rand(-maxd, maxd, 1)

        if num + 3 * delta > high:
            delta = -maxd
        elif num + 3 * delta < low:
            delta = maxd

        num += delta

        if mode == 0:
            num = round(num)
        
        list[i] = num
    
    return list

########

def rand_pos2(mode):
    # mode 0: all, mode 1: left, mode 2: right
    assert ((0<=mode<=2) == True), "rand_pos2 mode"

    if mode == 0:
        minx = -r
        maxx = r
    elif mode == 1:
        minx = -r
        maxx = -0.03
    else:
        minx = 0.03
        maxx = r
    
    x = rand(minx, maxx, 1)
    y = np.sqrt(r**2 - x**2) * rand(0.05, 0.95, 1) + r

    return [x, y]

def in_range(x, y, mode):
    # mode 0: all, mode 1: left, mode 2: right
    assert ((0<=mode<=2) == True), "in_range mode"

    if mode == 0:
        minx = -r
        maxx = r
    elif mode == 1:
        minx = -r
        maxx = -0.03
    else:
        minx = 0.03
        maxx = r

    if ((minx < x < maxx) == False):
        return False
    
    return ((x**2 + (y-r)**2) < r**2)

def vector_move(pos_i, speed, angle):
    pos_f = []
    pos_f.append(pos_i[0])
    pos_f.append(pos_i[1])
    dx = speed * np.cos(angle)
    dy = speed * np.sin(angle)
    pos_f[0] += dx
    pos_f[1] += dy
    return pos_f

def get_angle_back(pos_i, pos_f):
    dx = pos_f[0] - pos_i[0]
    dy = pos_f[1] - pos_i[1]

    if dx == 0:
        if dy > 0:
            return np.pi/2
        else:
            return 3*np.pi/2

    angle = np.arctan(dy/dx)

    if dx < 0:
        angle += np.pi
    if angle < 0:
        angle = np.pi * 2 + angle

    return angle

def rand_moving_pos(length, mode):
    # mode 0: all, mode 1: left, mode 2: right
    assert ((0<=mode<=2) == True), "rand_moving_pos mode"

    list = np.empty((length, 2), dtype=float)
    speed = rand(0,0.02,1)
    angle = rand(0,2*np.pi,1)
    home = [[0,r], [-0.5*r, r], [0.5*r, r]]

    pos = rand_pos2(mode)
    step = [8, 4, 4]

    for i in range(0, length):
        if (i % step[mode] == 0):
            speed = rand(0,0.02,1)
            angle = rand(0,2*np.pi,1)

        pos_next = vector_move(pos, step[mode] * speed, angle)

        if (in_range(pos_next[0], pos_next[1], mode) == False):
            angle = get_angle_back(pos_next, home[mode])

        pos = vector_move(pos, speed, angle)
        list[i] = np.array(pos)

    return list

########

def gen_data(fetus_num, pos_var, f_var):
    master = [0 for x in range(16)]

    # S0, S1, S2 magnitudes
    master[8] = rand(8, 10, 1)
    master[11] = rand(0.8, 1.0, 1)
    master[0] = np.ones(allt, dtype=float) * master[8]
    master[3] = np.ones(allt, dtype=float) * master[11]
    if fetus_num == 1:
        master[14] = 0
        master[6] = np.zeros(allt, dtype=float)
    else:
        master[14] = rand(0.8, 1.0, 1)
        master[6] = np.ones(allt, dtype=float) * master[14]

    # position
    if pos_var == True:
        if fetus_num == 1:
            master[10] = "time-varying"
            master[13] = "N/A"
            master[2] = rand_moving_pos(allt, 0)
            master[5] = np.zeros((allt, 2))
        else:
            master[10] = "time-varying"
            master[13] = "time-varying"
            master[2] = rand_moving_pos(allt, 1)
            master[5] = rand_moving_pos(allt, 2)
        
    else:
        if fetus_num == 1:
            master[10] = rand_pos(0)
            master[13] = "N/A"
            master[2] = np.zeros((allt, 2))
            master[2][:,:] = np.array(master[10])
            master[5] = np.zeros((allt, 2))
        else:
            master[10] = rand_pos(1)
            master[13] = rand_pos(2)
            master[2] = np.zeros((allt, 2))
            master[2][:,:] = np.array(master[10])
            master[5] = np.zeros((allt, 2))
            master[5][:,:] = np.array(master[13])

    # frequency
    if f_var == True:
        if fetus_num == 1:
            master[9] = "time-varying"
            master[12] = "time-varying"
            master[15] = "N/A"
            master[1] = rand_moving(60, 200, 10, allt, 0)
            master[4] = rand_moving(60, 200, 10, allt, 0)
            master[7] = np.ones(allt, dtype=float)
        else:
            master[9] = "time-varying"
            master[12] = "time-varying"
            master[15] = "time-varying"
            master[1] = rand_moving(60, 200, 10, allt, 0)
            master[4] = rand_moving(60, 200, 10, allt, 0)
            master[7] = rand_moving(60, 200, 10, allt, 0)
    else:
        if fetus_num == 1:
            master[9] = rand(60, 200, 0)
            master[12] = rand(60, 200, 0)
            master[15] = rand(60, 200, 0)
            master[1] = np.ones(allt, dtype=float) * master[9]
            master[4] = np.ones(allt, dtype=float) * master[12]
            master[7] = np.ones(allt, dtype=float) * master[15]
        else:
            master[9] = rand(60, 200, 0)
            master[12] = rand(60, 200, 0)
            master[15] = rand(60, 200, 0)
            master[1] = np.ones(allt, dtype=float) * master[9]
            master[4] = np.ones(allt, dtype=float) * master[12]
            master[7] = np.ones(allt, dtype=float) * master[15]

    return master

# Heading: 6. Execution

note = input("\nNote? ")
f_name = input("Log file name? ")
f = open(f"{f_name}.txt", "a")

int_no = int(input("Number of iteration? "))
fetus_num = int(input("Number of fetus? (1/2) "))
pos_var = int(input("Time-varying position? (0: False, 1: True) ")) == 1
f_var = int(input("Time-varying frequency? (0: False, 1: True) ")) == 1

assert (int_no >= 1), "Invalid number of iteration"
assert (fetus_num == 1) or (fetus_num == 2), "Invalid number of fetus"

input("Confirm begin? ")

accuracy = 0
error = 0.0
tot_count = 0

f.writelines("\n########\n")
f.writelines("Begin!\n")
f.writelines("########\n\n\n")
print("\n########")
print("Begin!")
print("########\n\n")
tstart = time.perf_counter()
dstart = datetime.datetime.now()

for i in range(0,int_no):

    tstarti = time.perf_counter()

    data = gen_data(fetus_num, pos_var, f_var)

    if pos_var == False:
        data[10] = f"({data[10][0]:.5f}, {data[10][1]:.5f})"
        if fetus_num == 2:
            data[13] = f"({data[13][0]:.5f}, {data[13][1]:.5f})"

    f.writelines(f"\t{i+1}/{int_no}\n")
    f.writelines(f"\tS1pos: {data[10]} S1m: {data[11]:.5f} S1f: {data[12]}\n")
    print(f"\t{i+1}/{int_no}")
    print(f"\tS1pos: {data[10]} S1m: {data[11]:.5f} S1f: {data[12]}")
    if fetus_num == 2:
        f.writelines(f"\tS2pos: {data[13]} S2m: {data[14]:.5f} S2f: {data[15]}\n")
        print(f"\tS2pos: {data[13]} S2m: {data[14]:.5f} S2f: {data[15]}")

    S0 = Source(np.zeros((allt, 2), dtype=float), data[0], data[1])
    S1 = Source(data[2], data[3], data[4])
    S2 = Source(data[5], data[6], data[7])

    Sm = SensorM(S0.value)
    Sa = Sensor(S_pos[0,0],S_pos[0,1],Sm.value, S1.pos, S1.value, S2.pos, S2.value)
    Sb = Sensor(S_pos[1,0],S_pos[1,1],Sm.value, S1.pos, S1.value, S2.pos, S2.value)
    Sc = Sensor(S_pos[2,0],S_pos[2,1],Sm.value, S1.pos, S1.value, S2.pos, S2.value)

    Sa.freq_domain = get_freq_domain(Sa.value - Sm.value)
    Sb.freq_domain = get_freq_domain(Sb.value - Sm.value)
    Sc.freq_domain = get_freq_domain(Sc.value - Sm.value)

    results = Results(Sa, Sb, Sc, S1, S2)
    results.compute()
    results.to_bpm()
    results.get_accuracy()

    accuracy += results.accuracy
    error += results.error
    tot_count += results.tot_count

    tendi = time.perf_counter()
    f.writelines(f"\tAccuracy: {results.accuracy:.0f} {results.accuracy/results.tot_count*100:.2f}% | Error: {results.error:.2f} ave={results.error/results.tot_count:.2f} (bpm) | Tot: {results.tot_count} | Time: {tendi-tstarti:.5f} (s)\n\n")
    print(f"\tAccuracy: {results.accuracy:.0f} {results.accuracy/results.tot_count*100:.2f}% | Error: {results.error:.2f} ave={results.error/results.tot_count:.2f} (bpm) | Tot: {results.tot_count} | Time: {tendi-tstarti:.5f} (s)")
    print(f"\tAccumulated Accuracy: {accuracy:.0f} {accuracy/tot_count*100:.2f}% | Error: {error:.2f} ave={error/tot_count:.2f} (bpm) | Tot: {tot_count}\n")

tend = time.perf_counter()
dend = datetime.datetime.now()

f.writelines("\n########\n")
f.writelines("Completed!\n")
f.writelines(f"Note: {note}\n")
f.writelines(f"Log: {f_name}.txt\n")
f.writelines(f"Iteration: {int_no} Equivalent to: {int_no/60:.2f} (h)  | Fetus: {fetus_num} | Time-varying position: {pos_var} | Time-varying frequency: {f_var}\n")
f.writelines(f"Accuracy: {accuracy:.0f} {accuracy/tot_count*100:.5f}% | Error: {error:.5f} ave={error/tot_count:.5f} (bpm) | Tot: {tot_count}\n")
f.writelines(f"Time taken: {tend-tstart:.5f} (s) | Start: {dstart} | End: {dend}\n")
f.writelines("########\n")
f.close()

print("\n########")
print("Completed!")
print(f"Note: {note}")
print(f"Log: {f_name}.txt")
print(f"Iteration: {int_no} Equivalent to: {int_no/60:.2f} (h) | Fetus: {fetus_num} | Time-varying position: {pos_var} | Time-varying frequency: {f_var}")
print(f"Accuracy: {accuracy:.0f} {accuracy/tot_count*100:.5f}% | Error: {error:.5f} ave={error/tot_count:.5f} (bpm) | Tot: {tot_count}")
print(f"Time taken: {tend-tstart:.5f} (s) | Start: {dstart} | End: {dend}")
print("########\n")
