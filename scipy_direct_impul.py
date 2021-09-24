from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.dates as mdates
from scipy.optimize import fsolve
from scipy import interpolate

#############
#### Test cases
#############
# 1 : solution for one t0 given below
# 2 : graphic of the number of release with respect to t0
test_case = 2


#############
#### Parametres of the releases and discretisation
#############
period = 7.0 # ones release per week
#taulambda=12000.0 # mosquitoes by massive releases per hectar
taulambdafinal = 100.0# small releases per hectar
if test_case == 1:
    t_init = np.datetime64('2009-09-01') # beginning of the time interval
    t_final = np.datetime64('2021-02-13') # end of the time interval
    t0 = np.datetime64('2012-05-01')# beginning of the releases
if test_case == 2:
    t_init = np.datetime64('2009-09-01') # beginning of the time interval
    t_final = np.datetime64('2021-07-14') # end of the time interval
    t0_init = np.datetime64('2018-01-01')
    t0_end =  np.datetime64('2018-04-28')
    date_array = np.arange(t0_init,t0_end)
    #mod_date = 2 # modulo for the date
T = (t_final - t_init).astype(float) #1158.0 # total time of the datas (1159 jour de mesure)
NT=int(T) # time discretization

#############
#### Constant parametres of the system
#############
"""print('Mechanical control:')
Mech_Cont = input()
print('residual fertility:')
epsilon = input()
print('release by hectar:')
taulambda = input()"""


for Mech_Cont in [0.2]:#[0.0,0.2,0.4]:
    for epsilon in [0.012]:#[0.0,0.006,0.012]:
        for taulambda in [6000]:#[6000,12000]:
            print('#################################')
            print('Mechanical control:',Mech_Cont )
            print('residual fertility:',epsilon)
            print('release by hectar:',taulambda)
            print('#################################')

            beta=0.9 # more precised ?
            Kmax=20.0*10000.0
            K0=20*100.0 # article Legoff2019 : reprendre la valeur ?
            #epsilon = 0.012 # residual fertility
            Rthresh=4.0 # 10 mm (See Valdez)
            #Mech_Cont=0.4 # mechanical control (add to muA2 in the definition of muA2)
            rate_I=0.0 # migration
            MT = 20.0*taulambda # size of the relaeses on 20i hectars
            MT_final = 20.0*taulambdafinal # size of the releases on  20 hectars



            def phi_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                #y = [4.1957,10.3637,9.7792,3.7462]
                y = [0.0,4.1957,7.1395,10.8968,1.1068]
                tck = interpolate.splrep(x, y, s=0)
                return interpolate.splev(Temp_t, tck, der=0)

            def muA1_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                y = [0.0198,0.0177,0.0260,0.047,0.2999]
                tck = interpolate.splrep(x, y, s=0)
                return interpolate.splev(Temp_t, tck, der=0)


            def gama_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                y = [0.0286,0.0694,0.0962,0.1136,0.0813]
                tck = interpolate.splrep(x, y, s=0)
                return interpolate.splev(Temp_t, tck, der=0)

            def muM_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                y = [0.0449,0.0676,0.0722,0.0811,0.0937]
                return np.interp(Temp_t,x,y)

            def muF_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                y = [0.0353,0.0458,0.0453,0.0413,0.0693]
                tck = interpolate.splrep(x, y, s=0)
                return interpolate.splev(Temp_t, tck, der=0)

            def r_f(Temp_t):
                x = [15.0,20.0,25.0,30.0,35.0]
                y = [0.475,0.435,0.41,0.463,0.666]
                tck = interpolate.splrep(x, y, s=0)
                return interpolate.splev(Temp_t, tck, der=0)



            ### temporal dynamic to solve the ode
            def f_scipy(y,t,param,MT):
                Gama,MuA1,MuA2,MuF,MuM,Phi, R = param[:,int((NT-1)*t/T)]
                A,M,F,M_s = y
                dA=Phi*F-(Gama + MuA1 + MuA2*A)*A
                dM=(1.0-R)*Gama*A-(MuM-rate_I/(1.0+M))*M
                dF=((M+epsilon*beta*M_s)/(M+beta*M_s))*R*Gama*A-(MuF-rate_I/(1.0*F))*F
                dM_s=-MuM*M_s
                return np.array([dA,dM,dF,dM_s ])



            ### compute the solution such that F(t1)<F_box
            def compute_sol(t0):
                # before the TIS
                if t0>0:
                    t_array_final=t_array[:int(NT*t0/T)+1]
                    sol = odeint(f_scipy, init,t_array_final, args=(param,MT))
                nb_release=0
                i = len(t_array)-1
                # beginning of the TIS
                while (sol[-1,0]>A_box_tab[i] or sol[-1,1]>M_box_tab[i] or sol[-1,2]>F_box_tab[i]) and int(NT*(t0+(nb_release+1)*period)/T)+1<NT:
                    init_temp=sol[-1]
                    init_temp[3]=init_temp[3]+MT
                    t_temp=t_array[int(NT*(t0+nb_release*period)/T):int(NT*(t0+(nb_release+1)*period)/T)+1]
                    sol_temp = odeint(f_scipy, init_temp,t_temp, args=(param,MT))
                    sol=np.concatenate((sol[:-1],sol_temp),axis=0)
                    t_array_final=np.append(t_array_final[:-1],t_temp)
                    i = len(t_array)-1
                    nb_release+=1

                nb_release_after = nb_release
                if int(NT*(t0+(nb_release+1)*period)/T)+1>=NT:
                    nb_release = -1

                # after the TIS when solution < F_box
                while t_array_final[-1]!=T:
                    init_temp=sol[-1]
                    init_temp[3]+=MT_final
                    t_temp=t_array[int(NT*(t0+nb_release_after*period)/T):min(int(NT*(t0+(nb_release_after+1)*period)/T)+1,NT+1)]
                    sol_temp = odeint(f_scipy, init_temp,t_temp, args=(param,MT))
                    sol=np.concatenate((sol[:-1],sol_temp),axis=0)
                    t_array_final=np.append(t_array_final[:-1],t_temp)
                    nb_release_after+=1
                if sol[-1,0]>A_box_tab[i] or sol[-1,1]>M_box_tab[i] or sol[-1,2]>F_box_tab[i]:
                    print("warning: out of the box")
                return sol,t_array_final,nb_release



            # importation data
            datas = np.loadtxt("meteor_La_Mare_01092009-14072021_serie_continue.txt")
            T_init_ind = t_init.astype(int) - np.datetime64('2009-09-01').astype(int) # index in the data corresponding to the beginning of the time interval
            T_final_ind = T_init_ind + int(T) # index in the data corresponding to the end of the time interval
            Rainfall=datas[T_init_ind:T_final_ind+1,4]#precipitation
            Temperature=datas[T_init_ind:T_final_ind+1,7] # temperature
            datas = np.loadtxt('meteorGilot01092009-14072021.txt')
            humidity=0.5*(datas[T_init_ind:T_final_ind+1,1]+datas[T_init_ind:T_final_ind+1,2]) #humidité relative à partir des données de la station météo de l'aéroport
            # valeur paramètre; voir papier Valduez 2018
            k_par=3.9e-5



            # construction of H and K
            H_tab=np.zeros(len(Temperature)+1)
            K_tab=np.zeros(len(Temperature))
            Hmax=max(Rainfall)
            for i in range(len(Temperature)):
                Delta=Rainfall[i]-(100.0-humidity[i])*k_par*(25.0+Temperature[i])**2
                if H_tab[i]+Delta<=0:
                    H_tab[i+1]=0.0
                else:
                    if H_tab[i]+Delta>=Hmax:
                        H_tab[i+1]=Hmax
                    else:
                        H_tab[i+1]=H_tab[i]+Delta
                K_tab[i]=Kmax*H_tab[i]/Hmax+K0



            #### interpolation de K and temperature on the new time discretization
            Temp_init=np.linspace(0,T,len(K_tab))
            t_array=np.linspace(0,T,int(NT)+1)
            K_tab = np.interp(t_array,Temp_init,K_tab)
            Temperature = np.interp(t_array,Temp_init,Temperature)


            ### Construction of the other time depending parameters
            phi_tab=np.zeros(int(NT+1))
            gama_tab=np.zeros(int(NT+1))
            muF_tab=np.zeros(int(NT+1))
            muM_tab=np.zeros(int(NT+1))
            muA1_tab=np.zeros(int(NT+1))
            muA2_tab=np.zeros(int(NT+1))
            r_tab=np.zeros(int(NT+1))
            BON_tab=np.zeros(int(NT+1))
            for i in range(int(NT+1)):
                phi_tab[i]=phi_f(Temperature[i])
                gama_tab[i]=gama_f(Temperature[i])
                muF_tab[i]=muF_f(Temperature[i])
                muM_tab[i]=muM_f(Temperature[i])
                muA1_tab[i]=muA1_f(Temperature[i])
                r_tab[i]=r_f(Temperature[i])
                muA2_tab[i]=r_tab[i]*gama_tab[i]*phi_tab[i]/((1.0-Mech_Cont)*K_tab[i]*muF_tab[i])
                BON_tab[i] = gama_tab[i]*r_tab[i]*phi_tab[i]/(muF_tab[i]*(gama_tab[i]+muA1_tab[i]))



            # computation of the equilibrium for us = 0
            BON=gama_tab*r_tab*phi_tab/(muF_tab*(gama_tab+muA1_tab))
            Astar=(1.0-1.0/BON)*K_tab
            Mstar=(1.0-r_tab)*gama_tab*Astar/muM_tab
            Fstar=r_tab*gama_tab*Astar/muF_tab
            print('Astar=',Astar[0])
            print('Fstar=',Fstar[0])
            print('Mstar=',Mstar[0])
            print('MT=',MT)
            Msstar=0.0
            print("Verif. eq. Astar:",max(abs(phi_tab*Fstar-(gama_tab + muA1_tab + muA2_tab*Astar)*Astar)))
            print("Verif. eq. Mstar:",max(abs((1.0-r_tab)*gama_tab*Astar-(muM_tab-rate_I/(1.0+Mstar))*Mstar)))
            print("Verif. eq. Fstar:",max(abs(((Mstar+epsilon*beta*Msstar)/(Mstar+beta*Msstar))*r_tab*gama_tab*Astar-(muF_tab-rate_I/(1.0*Fstar))*Fstar)))




            # computation of the equilibrium for us = MT_final
            Ms1=MT_final*np.exp(-muM_tab*period)/(1.0-np.exp(-muM_tab*period))
            Q=gama_tab*(1.0-r_tab)*(gama_tab+muA1_tab)/(muA2_tab*muM_tab)
            Delta_eps = (Q*(BON-1.0)-beta*Ms1)**2-4.0*Q*beta*Ms1*(1.0-epsilon*BON)
            A_box_tab=np.zeros(int(NT+1))
            M_box_tab=np.zeros(int(NT+1))
            F_box_tab=np.zeros(int(NT+1))
            for i in range(int(NT+1)):
                if Delta_eps[i]<0:
                    A_box_tab[i], M_box_tab[i], F_box_tab[i] = 10000.0, 10000.0, 10000.0
                else:
                    A1 = muM_tab[i]*(Q[i]*(BON[i]-1.0)-beta*Ms1[i]-np.sqrt(Delta_eps[i]))/(2.0*(1.0-r_tab[i])*gama_tab[i])
                    A2 = muM_tab[i]*(Q[i]*(BON[i]-1.0)-beta*Ms1[i]+np.sqrt(Delta_eps[i]))/(2.0*(1.0-r_tab[i])*gama_tab[i])
                    F1=(gama_tab[i]+muA1_tab[i]+muA2_tab[i]*A1)*A1/phi_tab[i]
                    F2=(gama_tab[i]+muA1_tab[i]+muA2_tab[i]*A2)*A2/phi_tab[i]
                    M1=(1.0-r_tab[i])*gama_tab[i]*A1/muM_tab[i]
                    M2=(1.0-r_tab[i])*gama_tab[i]*A2/muM_tab[i]
                    if epsilon*BON[i]>1.0:
                        A_box_tab[i], M_box_tab[i], F_box_tab[i] = A2, M2, F2
                    else:
                        MT1 = Q[i]*(BON[i]+1.0-2.0*epsilon*BON[i]-2.0*np.sqrt((1.0-epsilon*BON[i])*(1.0-epsilon)*BON[i]))/beta
                        if Ms1[i]>MT1:
                            A_box_tab[i], M_box_tab[i], F_box_tab[i] = 10000.0, 10000.0, 10000.0
                        else:
                            A_box_tab[i], M_box_tab[i], F_box_tab[i] = A1, M1, F1
            print("Verif. eq. A1:",max(abs((phi_tab*F_box_tab-(gama_tab + muA1_tab + muA2_tab*A_box_tab)*A_box_tab))))
            print("Verif. eq. M1:",max(abs(((1.0-r_tab)*gama_tab*A_box_tab-(muM_tab-rate_I/(1.0+M_box_tab))*M_box_tab))))
            print("Verif. eq. F1:",max(abs((((M_box_tab+epsilon*beta*Ms1)/(M_box_tab+beta*Ms1))*r_tab*gama_tab*A_box_tab-(muF_tab-rate_I/(1.0+F_box_tab))*F_box_tab))))


            #print(min(A_box_tab),min(M_box_tab),min(F_box_tab))
            #print(max(A_box_tab),max(M_box_tab),max(F_box_tab))
            #print(min(BON))

            # remplacement of the time dependent box by the min of the box
            A_box_tab = min(A_box_tab)*np.ones(len(A_box_tab))
            M_box_tab = min(M_box_tab)*np.ones(len(M_box_tab))
            F_box_tab = min(F_box_tab)*np.ones(len(F_box_tab))


            ### Initial condition and parameters
            init = [Astar[0],Fstar[0],Mstar[0],0.0]
            param=np.array([gama_tab,muA1_tab,muA2_tab,muF_tab,muM_tab,phi_tab,r_tab])




            if test_case ==1:
                t0=t0.astype(int) - t_init.astype(int)
                print('t0=',t0)
                date_array = np.arange(t_init, t_final+1)

                # computation of the solution
                sol,t_array_final,nb_release = compute_sol(t0)
                A_array=sol[:,0]
                M_array=sol[:,1]
                F_array=sol[:,2]
                M_s_array=sol[:,3]
                print('max |A|: ',max(abs(A_array)))
                print('max |M|: ',max(abs(M_array)))
                print('max |F|: ',max(abs(F_array)))
                print('max |M_s|: ',max(abs(M_s_array)))

                print("nb_release",nb_release)
                # plot K and temperature to see if it is smooth enough
                plt.figure(1)
                plt.subplot(8,1,1)
                plt.plot(t_array,K_tab,label='K')
                plt.legend(loc='best')
                plt.subplot(8,1,2)
                plt.plot(t_array,Temperature,label='Temperature')
                plt.legend(loc='best')
                plt.subplot(8,1,3)
                plt.plot(t_array,phi_tab,label='phi')
                plt.legend(loc='best')
                plt.subplot(8,1,4)
                plt.plot(t_array,gama_tab,label='gamma')
                plt.legend(loc='best')
                plt.subplot(8,1,5)
                plt.plot(t_array,muF_tab,label='muF')
                plt.legend(loc='best')
                plt.subplot(8,1,6)
                plt.plot(t_array,muM_tab,label='muM')
                plt.legend(loc='best')
                plt.subplot(8,1,7)
                plt.plot(t_array,muA1_tab,label='muA1')
                plt.legend(loc='best')
                plt.subplot(8,1,8)
                plt.plot(t_array,muA2_tab,label='muA2')
                plt.legend(loc='best')
                plt.savefig("numpy_direct_tis_parameters.png")


                # Plot solution
                fig, ax = plt.subplots(figsize = (10,6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.plot(date_array.astype('O'),A_array,'r-',linewidth=2,label=r'$A$')
                ax.plot(date_array.astype('O'),M_array,'g-',linewidth=2,label=r'$M$')
                ax.plot(date_array.astype('O'),F_array,'b-',linewidth=2,label=r'$F$')
                ax.plot(date_array.astype('O'),M_s_array,'-',linewidth=2,label=r'$M_s$')
                cond = np.mod(range(len(date_array)),int(T/20.0))==True # plot the date every T/20 days
                ax.set_xticks(np.extract(cond,date_array.astype('O')))
                plt.legend(loc='best')
                plt.xlabel('Time')
                fig.autofmt_xdate()
                plt.savefig("scipy_direct_tis_solution_MT_{name0}.png".format(name0 = str(MT)))
                plt.show()


            if test_case == 2:
                diff = date_array[0] - t_init
                t0_min = diff.astype(int)
                print('t0 min=',t0_min)
                diff = date_array[-1] - t_init
                t0_max = diff.astype(int)
                print('t0 max=',t0_max)
                t0_array=range(t0_min,t0_max+1)
                #cond = np.mod(t0_array,mod_date)==True
                #date_array =np.extract(cond ,date_array)
                #t0_array = np.extract(cond,t0_array)
                nb_release_array=np.zeros(len(t0_array))
                for i_t0 in range(len(t0_array)):
                    t0 = float(t0_array[i_t0])
                    print('t0=',t0)
                    sol,t_array_final,nb_release = compute_sol(t0)
                    nb_release_array[i_t0] = nb_release
                #plt.clf()
                #fig, ax = plt.subplots()
                #ax.plot(date_array.astype('O'), nb_release_array)
                #fig.autofmt_xdate()
                #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                #cond = np.mod(range(len(date_array)),int(float(len(date_array))/10.0))==True # plot the date every (len(date_array))/20 days
                #ax.set_xticks(np.extract(cond,date_array.astype('O')))
                #ax.set_title('Number of releases (=0 if not enough time) with restpect to t0')
                #plt.grid()
                #plt.savefig("scipy_direct_tis_nb_release_t0min{name0}_t0max{name1}_MC{name2}_MT{name3}_RF{name4}.png".format(name0=t0_init,name1=t0_end,name2=int(100.0*Mech_Cont),name3=MT,name4=int(1000.0*epsilon)))
                np.savetxt("nb_releases_t0min{name0}_t0max{name1}_MC{name2}_MT{name3}_RF{name4}.out".format(name0=t0_init,name1=t0_end,name2=int(100.0*Mech_Cont),name3=MT,name4=int(1000.0*epsilon)),np.transpose([date_array.astype(str),nb_release_array.astype(int)]),fmt='%s')
                #plt.show()

