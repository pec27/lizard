Hi, once you have downloaded Lizard you will probably want to do the following:

1) make # builds the C-functions

2) run the tests, e.g. with
nosetests tests/
or
nosetests -v tests/
or
nosetests tests/test_p3m.py # for a specific file

3) Add Lizard to your PYTHONPATH, e.g. 
PYTHONPATH="${PYTHONPATH}:${HOME}/codes/lizard"
export PYTHONPATH

Which you might want to put in your .profile. 

NOTE that if you plan to experiment with your own modifications, you probably 
want to make sure '.' appears first on your PYTHONPATH, otherwise when 
checking out lizard in another directory (say lizard_my_experiments) you may 
find nosetests picks up the old lizard to test.
