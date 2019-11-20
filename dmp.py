
import numpy as np

from scipy.integrate import ode

class DMP:
    def __init__(self,n_dmp_basis,num_steps,n_actuators):
        self.n_dmp_basis = n_dmp_basis# = 10
        self.num_steps = num_steps
        self.n_actuators = n_actuators

    def basis_function(self,t,t0):
        # return np.exp(-0.5*(t-t0)**2/((n_simulation_steps/n_dmp_basis)**2))
        return np.exp(-0.5*(t-t0)**2/((1.0/self.n_dmp_basis)**2))

    def basis_functions(self,t,x,g,w,y0):
        # phis = np.array(list(map(lambda t0: basis_function(t,t0), np.linspace(0,n_simulation_steps,n_dmp_basis)))).T
        phis = np.array(list(map(lambda t0: self.basis_function(t,t0), np.linspace(0,1.0,self.n_dmp_basis)))).T
        return x*(g-y0)*np.matmul(w,phis)/np.sum(phis)

    def dmp(self,t, variables, w, g):
        n_actuators = self.n_actuators
        y0 = np.zeros(n_actuators)
        alphay = 2.0
        betay = 1.0
        # print(w)
        alphax = 0.5
        variables = variables.reshape((n_actuators,3))
        y,v,x = variables[:,0],variables[:,1],variables[:,2]
        # print(y.shape, v.shape, x.shape, g)
        vdot = alphay*(betay*(g-y)-v) + 5e2*self.basis_functions(t,x,g,w.reshape((self.n_actuators,self.n_dmp_basis)),y0)
        ydot = v
        xdot = -alphax*x
        return np.stack([ydot,vdot,xdot],axis=1).reshape((n_actuators*3))

    def action_rollout(self,context,action_parameter,i):
        n_actuators = self.n_actuators
        dt = 1.0/self.num_steps
        if i==0:
            solver = ode(self.dmp)
            # print(action_parameter)
            solver.set_initial_value(np.tile(np.array([0,0,1]),(n_actuators,1)).reshape(-1),0)\
                .set_f_params(action_parameter[:-n_actuators],action_parameter[-n_actuators:])
            self.solver = solver
            return np.clip(self.solver.integrate(self.solver.t+dt).reshape((n_actuators,3))[:,0], -1,1)
        else:
            return np.clip(self.solver.integrate(self.solver.t+dt).reshape((n_actuators,3))[:,0], -1,1)
