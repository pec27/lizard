"""
Module for a log class which prints to files+stdout+timestamps

Should probably extend this to a general I/O module as I proceed.

"""
from __future__ import print_function, unicode_literals
from time import time
import sys
# Ugly Python 2 and 3 buffering
try:
    from cStringIO import StringIO
except:
    from io import StringIO 

        

class VerboseTimingLog:
    def __init__(self, bufs=None, filename=None, also_stdout=True, insert_timings=True):
        """
        filename       - [None] Optional file to write to
        also_stdout    - [True] Optionally log to stdout (in addition)
        insert_timings - [True] to decorate with time stamps

        The timeing works by catching each write that ends with a newline. If 
        you write one, e.g. 
        >>> print('Doing FFT', file=log)
        >>> fft()
        >>> print('Done', file=log)

        then the string excluding the newline is printed, and the timer runs
        until the next write arrives, at which point the time (e.g. '[1.54s]')
        is appended to old line and then the new line is printed, i.e. you get 
        the output:

        Doing FFT [1.54s]
        Done

        """
        logs = []
        if filename is not None:
            self._logfile = open(filename, 'a')
            logs.append(self._logfile)
        else:
            self._logfile = None

        if bufs is not None:
            logs.extend(bufs)

        if also_stdout:
            logs.append(sys.stdout)
        self._logs = logs
        self._timings = insert_timings
        self._last_time = time()
        self._ending_lap = False # [dont] hold a carriage return for timer

    def _all_write_flush(self, text):
        """ write and flush to all logs """
        for log in self._logs:
            log.write(text)
            log.flush()
    
    def write(self, text):
        """
        Parse the text into the log file(s), decorating with time stamps if desired.

        As noted in __init__, a string that terminates with a new line indicates that
        the next interval should be timed.
        """
        # In case the input was an iterator, will extract
        buf = str(text)

        if len(buf)==0 or not self._timings:
            return self._all_write_flush(buf) # just pass through
        
        # If the last character was a newline, next thing is always a timestamp
        if self._ending_lap:
            self._write_time_stamp()

        # If we end with a newline, (re)start timer
        self._ending_lap = (buf[-1]=='\n')
        if self._ending_lap:
            # Write the line but drop the newline, since we are going to put a 
            # timestamp there when the calculation completes
            self._all_write_flush(buf[:-1])
            self._all_write_flush(' ')  # going to follow with timestamp

        else:
            self._all_write_flush(buf) # Just write the buffer

    def close(self):
        # Put in remaining time stamps
        if self._ending_lap:
            self._write_time_stamp()
            self._ending_lap = False
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None

    def _write_time_stamp(self):
        # append the time stamp to previous line
        new_time = time()
        delta = new_time - self._last_time
        self._last_time = new_time
        time_stamp = '[%.2fs]\n'%delta
        self._all_write_flush(time_stamp)
        
    def __del__(self):
        """ Make sure logfile is closed """
        self.close()


class PrefixedIO:
    """
    Write prefix, e.g. '[Proc 15] ' before every write.
    """
    def __init__(self, buf, prefix):
        self._prefix = str(prefix)
        self._buf = buf
        self._newline = True # starting a new line

    def write(self, txt):
        s = str(txt)

        while len(s):
            if self._newline:
                self._buf.write(self._prefix)
            self._newline = '\n' in s #Always newline after carriage return
            if not self._newline:
                return self._buf.write(s) # just write out buffer

            line_tail, cr, s = s.partition('\n') 
            self._buf.write(line_tail+cr) # write up to CR
    def flush(self):
        self._buf.flush()

class LockedLineIO:
    """
    Acquires a lock before writing lines (ending with carriage returns), caches 
    incomplete lines to write later. Useful for multiple processes writing to the
    same log.
    """
    def __init__(self, bufs, lock):
        self._bufs = bufs
        self._lock = lock
        self._buf = StringIO() # store incomplete lines

    def write(self, txt):
        """ write (whole lines only) """
        s = str(txt)
        if '\n' not in s:
            self._buf.write(s) # not a whole line
            return
        
        # Otherwise write out
        to_last_cr, cr, s = s.rpartition('\n')
        self._buf.write(to_last_cr+cr)
        lines_with_cr = self._buf.getvalue()

        # lock and write
        with self._lock: # Only 1 process can have it
            for buf in self._bufs:
                buf.write(lines_with_cr)
                buf.flush()
        self._buf = StringIO()
        # Remainder
        self._buf.write(s)
        
    def flush(self):
        return
    def __del__(self):
        s = self._buf.getvalue()
        if len(s)>0:
            self.write('!! Unterminated string\n') # we didn't finish!


class MarkUp:
    """
    Some markup to colour the text
    """
    ENDC = '\033[0m' # revert to normal text
    @classmethod
    def enable(cls):
        """ Globablly enable markup """
        cls.HEADER = '\033[95m'
        cls.OKBLUE = '\033[94m'
        cls.OKGREEN = '\033[92m'
        cls.WARNING = '\033[93m'
        cls.FAIL = '\033[91m'
        cls.BOLD = '\033[1m'
        cls.UNDERLINE = '\033[4m'
    
    @classmethod
    def disable(cls):
        """ Globally disable markup """
        cls.HEADER = cls.OKBLUE = cls.OKGREEN = cls.WARNING = cls.FAIL = cls.BOLD = cls.UNDERLINE = cls.ENDC


# Markup on by default
MarkUp.enable()

# Default null log
class NullLog:
    def write(self, text):
        pass

null_log = NullLog()
