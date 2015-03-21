# python-sparse-category-indices
An implementation of "sparse category indices" from (Madani and Connor, 2008) - "Large-Scale Many-Class Learning" http://omadani.net/pubs/2008/many_class_SDM08.pdf.
-----------------------------------------------------
This software is licenced under the Apache Licence, Version 2.0.

To acquire a copy of this licence, visit the following
link:

http://www.apache.org/licenses/LICENSE-2.0.html
-----------------------------------------------------
This software has been tested under Python 2.6x. It
may not work under other 2.x versions of the Python interpreter.

You need to have 'hotshot' installed for profiling.

For options, run [from this directory]:

python ./runff.py -h

All of the magic happens in ./ffclassifier.py

To make this work, you need to write a feature extractor that
writes features of the form:

<class> <featname1>(:<activation1>) ... <featnameN>(:<activationN>)

to file, where <class> is an output class string, <featnameX> are 
string feature names and <activationX> are real-valued feature 
activations (not needed for Boolean features).

Here's an example of how to run the classifier (using the toy 
program 'lm.py' -- an example discriminative language modelling
use):

-----------------------------------------------------
$ wget http://norvig.com/big.txt
$ head -10000 big.txt | sed 's/\([.,?\!\";:]\)/ \1 /g' | python ./lm.py --train --context=2 > /tmp/fts
$ python runff.py -m /tmp/mod -f /tmp/fts -b -t -i 2
Training on data set /tmp/fts: iteration 1
MRR for iteration 1: 0.234908631699
R-1 recall for iteration 1: 0.189187471472
R-5 recall for iteration 1: 0.292782980985
Training on data set /tmp/fts: iteration 2
MRR for iteration 2: 0.876870852508
R-1 recall for iteration 2: 0.785523206434
R-5 recall for iteration 2: 0.989831115631
total time in seconds 69
ave. per-iteration time in seconds 34
$ echo "elementary _ dear Watson" | python ./lm.py --test --model /tmp/mod --context=2
elementary {my} dear Watson
-----------------------------------------------------

