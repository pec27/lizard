"""
Kick-drift-kick integration and Cosmological integration using P^3M.

Peter Creasey - Jan 2016
"""
from __future__ import print_function, absolute_import, division, unicode_literals
import lizard.p3m as p3m
from lizard.log import MarkUp as MU, null_log
from numpy import square, floor
from time import time

def get_pos_vel(model):
    """
    return pos,vel from the model 
    (default export function of the integrator)
    """
    # TODO be careful when we start to re-order particles
    pos, wts, vel = model.get_pos_wts_vel()
    return pos, vel


def kdk_integrate(model, export_times, log, exporter=get_pos_vel):
    """
    Integrate the model from t0 to t1 using the kick-drift-kick method


    In summary:
    export_times - float times to export at
    model - an object with functions
        model.get_time()        - current time of the model
        model.update(kicker)    - where kicker(lims, vel, acc) 
                                  returns (dt, dx, dvel)
        model.write_steps(lims) - Write the step limits lims in model units

    log   - log file
    [exporter=get_pos_vel - function to call on the model at each 
                integration output time, default returns pos and vel]

    
    returns None [model has had update method called though]

    Implementation is a via dependency injection, i.e. model needs to have a
    function model.update whose argument is the 'kick' call-back function. This
    should be called by model.update with its estimation for timestep limits (a 
    tuple of (dt, name) pairs) and the current x, dx/dt and d^2/dt^2. The 
    integrator then chooses a timestep and returns to model.update with
    increments (delta t, delta x, delta dx/dt) with which the model should 
    update itself. 

    This method was chosen so that the basic gravitational force in a 
    GravityModel could be wrapped with the cosmological expansion terms (see
    power.py).
    
    """


    # Make sure we have a list we can pop from
    times_left = [float(t1) for t1 in export_times] 

    step = 0
    t_begin = time()
    while len(times_left)>0:
        print(MU.BOLD + '-'*70 + MU.ENDC, file=log)
        t_start_step = time()

        print(MU.HEADER+'Step {:,}'.format(step)+MU.ENDC,file=log)
        t_cur = model.get_time()
        # timestep for next snapshot
        t1 = times_left[0]
        rec_dt = [(t1 - t_cur,'Next snapshot')]

        # build a kick function
        def kdk_kicker(dts, vel, accel):
            """
            Do the Kick-drift-kick
            """
            rec_dt.extend(dts) # record timestep limits
            # minimum timestep
            dt = min(dt for dt,key in rec_dt)

            half_kick = (0.5 * dt) * accel
            vel1 = vel + half_kick
            delta_pos = vel1 * dt
            delta_vel = 2 * half_kick
        
            return dt, delta_pos, delta_vel

        model.update(kdk_kicker)

        dts, keys = zip(*rec_dt)
        key_dt = keys[dts.index(min(dts))]
        print('Step limiter:', key_dt, file=log)
        model.write_steps(rec_dt)

        if key_dt=='Next snapshot':
            # done
            t_cur = times_left.pop(0)
            yield exporter(model)

        step = step+1 
        print(MU.OKBLUE+'Step took %.3fs'%(time()-t_start_step)+MU.ENDC,
              file=log)

    total_time = time() - t_begin
    print('{:,} steps in'.format(step), '%.2f seconds (%.5f seconds per step)'%(total_time, total_time/step), file=log)
    print('-'*70, file=log)
    return

class SingleProcGravityModel:
    """
    Class to hold and apply accelerations for many particles. The idea is
    that other versions of this class can be made with multiprocessor 
    support

    """
    def __init__(self, r_split, eta=0.025, log=null_log):
        
        # Build the force-splitter
        self._fs = p3m.get_force_split(r_split, mode='cubic')
        self._pm_accel = p3m.IncrementalPMAccel(self._fs)
        self._eta = float(eta)
        self._log = log

    def get_time(self):
        return float(self._time)

    def set_time_pos_wts_vel(self, time, pos, wts, vel):
        if pos.min()<0 or pos.max()>=1:
            print('Pos in',pos.min(axis=0), pos.max(axis=0), file=self._log)
            raise Exception('Positions should be in [0,1)')

        self._time = time
        self._pos = pos
        self._wts = wts
        self._vel = vel

    def update(self, kicker, r_soft):
        """
        Updater function. The current time (t) is known at this point but
        dt is not, you have to pass your suggested dt into the kicker function
        along with the pos, vel and accel

        r_soft - the softening length (in unit box scale)
        """
        pos, wts, vel = self.get_particles()

        # short range force
        pairs, acc_short = p3m.pp_accel(self._fs, wts, pos, r_soft)

        # long range force
        acc_long = self._pm_accel.accel(wts, pos, vel, self._time, self._log)

        acc = acc_short + acc_long

        # Maximum |acc|
        max_acc = square(acc).sum(1).max()**0.5

        # Standard gadget timestep
        dt_lim = (2*self._eta*r_soft/max_acc)**0.5

        dts = ((dt_lim, 'Accel'),)
        dt, dpos, dvel = kicker(dts, vel, acc)
        pos = pos + dpos

        self._time = self._time + dt
        self._pos = pos - floor(pos)
        self._vel = vel + dvel

    def get_particles(self):
        return self._pos, self._wts, self._vel



class NonCosmologicalPeriodicParticleModel:
    """
    Wrapper for a gravity model without cosmological terms (mostly for testing)
    For simplicity, G=1.
    """
    def __init__(self, part_model, pos, wts, vel, r_soft, log=null_log):
        
        # Defensive...
        assert(hasattr(part_model, 'update'))
        assert(hasattr(part_model, 'get_particles'))
        assert(hasattr(log, 'write'))
        assert(hasattr(log, 'flush'))
        
        self._part_model = part_model
        self._log = log
        self._r_soft = float(r_soft)

        part_model.set_time_pos_wts_vel(0.0, pos, wts, vel)

    def get_time(self):
        return self._part_model.get_time()

    def get_pos_wts_vel(self):
        return self._part_model.get_particles()

    def update(self, kicker):
        # pass in the fixed softening
        return self._part_model.update(kicker, self._r_soft)

    def write_steps(self, steps):
        # No units, just print
        for dt, key in steps:
            print('%f (%s)'%(dt, key), file=self._log)
