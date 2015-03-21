"""
A classifier using the Feature Focus model from Madani and Connor (2007,2008).
Authors: Dennis N. Mehay and Chris H. Brew
"""


import sys, os, time
import doctest
from optparse import OptionParser
import StringIO, codecs

# try to import profiler. will crash if not there and profile option is set below.
try:
    import hotshot,hotshot.stats
except:
    pass

import bisect
#from collections import defaultdict

def recallAtK(cls, ranking, k):
    """
    This function takes a class label ('cls'), a ranked list 'ranking' of (class,score)
    pairs (ranked by virtue of being sorted in descending order of score) and
    a parameter 'k', and returns 1 if 'cls' is in 'ranking' in the top 'k' ranked
    classes.  The word 'recall' is for conformity with IR terminology.  Think of this
    as an indicator function saying whether 'cls' is "recalled" in the top 'k' classes
    of 'ranking'.

    @param cls: [the string name of the class to search for in the top k ranked classes in
    'ranking'.]
    @param ranking: [the ranked list of (class,score) pairs.]
    @param k: [the number of top ranked classes to search in for 'cls'.]
    @return: [1 if 'cls' is in the top 'k' slots of 'ranking' and 0 otherwise.]

    >>> ranking = [('a',1.5), ('b',0.45), ('c',0.22), ('d',0.1)]
    >>> recallAtK('c',ranking,1)
    0
    >>> recallAtK('c',ranking,2)
    0
    >>> recallAtK('c',ranking,3)
    1
    >>> recallAtK('c',ranking,4)
    1
    >>> recallAtK('a',ranking,1)
    1
    >>> recallAtK('a',ranking,4)
    1
    """
    indicator = 0
    rank = 0
    for (cl,score) in ranking:
        rank += 1
        if rank <= k:
            if (cl == cls):
                indicator = 1
                break
        else:
            break
    return indicator

def recipRank(cls, ranking):
    """
    This function takes a class and returns its reciprocal rank score in the
    sorted association list of scores [(cls1,score), ..., (clsN,score)] 'ranking'
    (sorted in descending order by score).
    Note that the reciprocal rank for classes not in the ranked list of scores is
    assumed to be 0.0.
    
    @param cls: [the class in question.]
    @param ranking: [the ranked list of scores [(cls1,score), ..., (clsN,score).]]
    @return: [the reciprocal rank of 'cls'.]

    >>> ranking = [('a',1.5), ('b',0.45), ('c',0.22), ('d',0.1)]
    >>> '%0.2f' % recipRank('c',ranking)
    '0.33'
    >>> '%0.1f' % recipRank('a',ranking)
    '1.0'
    >>> '%0.1f' % recipRank('b',ranking)
    '0.5'
    """
    rr = 0.0
    rank = 0
    for i in ranking:
        rank += 1
        if cls == i[0]:
            rr = 1./rank
            break
    return rr
    
def by_descending_count(pair):
    """
    This function acts as a key for sort.

    It helps sort pairs of (key,count) into descending order of count
    
    >>> x = [("c",1),("a",2),("b",0)]
    >>> x.sort(key=by_descending_count)
    >>> x
    [('a', 2), ('c', 1), ('b', 0)]
    """
    return -pair[1]


class FeatureFocusModel(dict):

    """
    A classifier using the FeatureFocus model from Madani and Connor (2007,2008).
    """
    slots = ["_dmax","_margin","_wmin"]
    def __init__(self, dmax=25, margin=0.5, wmin=0.01,items=[]):
        """
        Initialize the classifer

        >>> FeatureFocusModel(10,0.2,0.1,[("a","b")])
        <FeatureFocusModel10,0.200000,0.100000,[('a', 'b')])
        >>> FeatureFocusModel(10,0.2,0.1)
        <FeatureFocusModel10,0.200000,0.100000,[])
        
        @param dmax: [how many classes we will let each feature contribute to the scoring of.]
        @param margin: [how close to error driven updates are we (0.0 is error driven and 1.0 is
            always update, while 0.5, 0.2, etc. fall somewhere in between).]
        @param wmin: [what is the minimum proportion that a feature can predict a class with and
            still be kept around -- 0.0 means keep all feature-class associations around,
            while 1.0 means keep (nearly?) none.]
       """
    
        self._dmax = dmax
        self._margin = margin
        self._wmin = wmin
        if items:
            for (k,v) in items:
                self[k] = v

    def __repr__(self):
        """
        >>> FeatureFocusModel(10,0.2,0.1)
        <FeatureFocusModel10,0.200000,0.100000,[])
        """
        return "<FeatureFocusModel%d,%f,%f,%r)" % (
                                                   self._dmax,self._margin,self._wmin,
                                                   self.items()
                                                   )

    def getClasses(self, f, inputact=1):
        """
        Retrieve all dmax classes for feature f as an association list of pairs

        @param f: [a FeatureFocus]
        @param inputact: [how much has feature f been activated?]
        @return: [a list of pairs of classes and association strengths for feature f]
        """
        return f.getClasses(inputact, self._dmax)


    def predict(self, classContextList, boolean=True):
        cl = classContextList[0]
        ctxt = self.makeContext(classContextList[1:], boolean)
        ranking = self.rankedRetrieval(ctxt)
        return ranking
    
    def learnFrom(self, classContextList, boolean=True):
        """
        Process an instance. May trigger an update of the model if margin
        is insufficient.
        
        >>> ffm = FeatureFocusModel(2,0.0,0.1)
        >>> ccl = ['a','f1','f2','f3']
        >>> ranking = ffm.learnFrom(ccl)
        >>> ranking
        []
        >>> ccl = ['b', 'f1','f4']
        >>> ranking = ffm.learnFrom(ccl) 
        >>> len(ranking) == 1
        True
        >>> ranking[0][0]
        'a'
        
        @param classContextList: [a list of [class, feat1, feat2, ...] representing a learning instance]
        @return: a ranking (pre-learning) over classes [(class,score), ..., (class, score)]. 

        """
        cl = classContextList[0]
        ctxt = self.makeContext(classContextList[1:],boolean)
        ranking = self.rankedRetrieval(ctxt)
        marg = self.computeMargin(cl, ranking)


        # following code handles update of active features
        # if the margin is insufficient
        if marg <= self._margin:
            for f in ctxt:
                # look up the class associations for each active feature of this
                # context and potentially strengthen the weight btwn it and the current
                # class.
                try:
                    ffeat = self[f]
                except KeyError:
                    ffeat =  FeatureFocus()

                ffeat.update(cl, ctxt[f], self._wmin)
                self[f] = ffeat
                
        # return the ranking that was calculated prior to
        # any updating that may have occurred
        return ranking
    
    def rankedRetrieval(self, context):        
        """
        Returns a ranking over classes as list of pairs (class,score) in descending order
        of score.

        @param context: [the active features that 'fire' in this context]
        @return: [as described]
        
        >>> ffm = FeatureFocusModel(2,0.2,0.1)
        >>> rk = ffm.learnFrom(['a','f1','f2','f3'])
        >>> rk = ffm.learnFrom(['b','f1','f2','f4'])
        >>> ctxt = ffm.makeContext(['f1','f4'])
        >>> ranking = ffm.rankedRetrieval(ctxt)
        >>> # build up a ranking with string-based repr's of scores.
        >>> str_ranking = []
        >>> i = 0
        >>> while i < len(ranking): str_ranking.append( (ranking[i][0], ('%0.1f' % ranking[i][1])) ); i += 1
        >>> str_ranking
        [('b', '0.2'), ('a', '0.1')]
        """
        ranking = {}

        for f in context:
            
            fetch = self.get(f)
            
            # only update class ranking for features already seen
            if not(fetch is None):               
                for (c,w) in self.getClasses(fetch, context[f]):
                    # update the score for class c
                    ranking[c] = ranking.get(c,0.0) + w
        
        res = [(k,ranking[k]) for k in ranking]
        # sort descending
        res.sort(key=by_descending_count)
        return res
    
    def makeContext(self, str_ls, boolean=True, splitter=':'):
        """
        Turns a string-list repr of a context in to a Context object.
        
        >>> ffl = FeatureFocusModel(2,0.1,0.1);
        >>> ctxt_str = ['f1','f1','f2','f3','f1','f2'];
        >>> ffl.makeContext(ctxt_str)
        {'f1': 3, 'f2': 2, 'f3': 1}
        >>> ffl = FeatureFocusModel(2,0.1,0.1);
        >>> ctxt_str = ['f1:1.0','f1:0.8','f2:0.3','f3:0.9','f1:1.0','f2:0.7'];
        >>> d = ffl.makeContext(ctxt_str, False);
        >>> for k,v in d.items(): d[k] = '%0.1f' %d[k]
        >>> d
        {'f1': '2.8', 'f2': '1.0', 'f3': '0.9'}
        """
        ctxt = Context()

        if boolean:            
            for f in str_ls:
                ctxt.addBoolean(f)
            return ctxt
        else:
            for f in str_ls:
                split = f.rfind(splitter)
                # oops.  no colons.
                if split == -1:
                    raise "No colons in alleged weighted list"
                else:
                    name = f[:split]
                    val = float(f[split+1:])
                    ctxt.addWeighted(name, val)
            return ctxt
        
    def computeMargin(self, correct, ranking):
        """
        Compute the margin of error given the correct class and a ranking over
        classes.

        >>> ffl = FeatureFocusModel(2,0.1,0.1);
        >>> ranking = [('a',0.3), ('b',0.2), ('c',0.1)];
        >>> correct = 'a';
        >>> "%0.2f" %ffl.computeMargin(correct, ranking)
        '0.10'

        @param correct: [a string giving the correct class.]
        @param ranking: [a list of pairs (class, score) sorted in descending order of score.]
        @return: [a real-valued margin of correctness that is (score_correct) - (score_highest_incorrect)
        (note that it can be 0.0 or negative).]

        """
        if ranking == []:
            return 0.0
        if len(ranking) == 1 and ranking[0][0]==correct:
            return ranking[0][1]
        
        correct_score = None
        wrong_score = None

        # go through the list until we find the score of the correct
        # category and that of the higest-ranked negative category (if
        # they're in there).
        i = 0
        while i < len(ranking) and ((correct_score is None) or (wrong_score is None)):
            (c,s) = ranking[i]
            if (correct_score is None) and (c == correct):
                correct_score = s
            elif wrong_score is None:
                wrong_score = s
            i += 1
            
        if correct_score is None:
            correct_score = 0.0
        if wrong_score is None:
            wrong_score = 0.0
            
        return correct_score - wrong_score
    
    def writeToFile(self, fname):
        """
        Write the current model parameters to file:

        @param fname: the name of the file to write the model to.
        @return: [None].
        """
        f = codecs.open(fname, "wb", "utf-8")

        try:
            print >> f, self._dmax, self._margin, self._wmin
        
            # the order in which items are presented in a dictionary
            # is not determined, but we want order to be definite so
            # that we can check file identity with diff,
            # so sort the items.
            its = self.items()
            its.sort()
            for feat_name, ffmap in its:
                ffits = ffmap.items()
                # if there are no items, we don't want to bother printing
                if ffits:            
                    # same reasoning as above: make item order canonical
                    ffits.sort()
                    print >> f, feat_name, ffmap._wtotal,                 

                    for classname,assoc in ffits:
                        print >> f, classname+':'+str(assoc),
                    f.write("\n")
        finally:
            f.close()

    def readFromFile(self, fname):
        self.clear() # I'm a kind of dictionary, and my contents are to be overwritten
        f = codecs.open(fname, "rb", "utf-8")
        try:
            (dmax,margin,wmin) = f.readline().split()
            self._dmax = int(dmax)
            self._margin = float(margin)
            self._wmin = float(wmin)

            for line in f:
                x = line.split()
                fname = x[0]
                wtotal = float(x[1])
                ff = FeatureFocus()
                for kv in x[2:]:
                    # fixed case where there was a colon in a substring of 'k' in the string 'kv' to be split.
                    splitpt = kv.rfind(":")
                    k = kv[:splitpt]
                    v = kv[splitpt+1:]
                    # with real-valued feature activation, 'v' can be a float.
                    ff[k] = float(v)
                    ff._wtotal = wtotal
                self[fname] = ff
        finally:
            f.close()
            
class Context(dict):
    """
    A Context is just a dictionary with two extra methods. This is a new-style
    Python class, because it inherits from dict.
    """
    __slots__ = []
    def addBoolean(self, f):
        """
        @param f: [the boolean feature that is activated.]
        @return: [None.]
        
        >>> ctx = Context()
        >>> ctx.addBoolean("tom")
        >>> ctx
        {'tom': 1}
        """
        self[f] = (self.get(f,0) + 1)

    def addWeighted(self, f, w):
        """
        @param f: [the real-valued feature that is activated.]
        @param w: [the real value of the feature.]
        @return: [None.]
        
        >>> ctx = Context()
        >>> ctx.addWeighted("jerry",2.0)
        >>> ctx
        {'jerry': 2.0}
        """
        self[f] = (self.get(f,0.0) + w)


    
class FeatureFocus(dict):
    """
    A 'smart' container that has a mapping from classes to association weights.
    this container can absorb updated associations, adjust its internals and drop
    those associations that fall below min. This is a new-style extension of dict.


    Try to adjust this to use two arrays rather than a dictionary.
    """
    __slots__ = ['_wtotal']
    def __init__(self):
        self._wtotal = 0.0

        

    def __str__(self):
        """
        @return: [a string representation of this FeatureFocus object.]
        """
        res = "wtot: "+str(self._wtotal)+"\n"
        for w in self.items():
            res += "   "+str(w)+"\n"
        return res.rstrip()

    def update(self, cls, inputact, wmin):
        """
        @param cls: [the class that was observed with input activation 'inputact' (1 for boolean features)].
        @param inputact: [the input activation strength of the feature.]
        @param wmin: [the minimum proportion that a feature can predict this class and still be kept
        in this focus set.]
        @return: None        

        >>> wmin = 0.10
        >>> ff = FeatureFocus()
        >>> ff.update('a',1,wmin)
        >>> ff.update('b',1,wmin)
        >>> cs = ff.getClasses()
        >>> i = 0;
        >>> str_cls = []
        >>> # turn each score into a string, formatted just so.        
        >>> while i < len(cs): str_cls.append( (cs[i][0],('%0.1f' % cs[i][1])) ); i+=1;
        >>> str_cls
        [('a', '0.1'), ('b', '0.1')]
        >>> ff.update('a',17,wmin)
        >>> ff.update('b',1,wmin)
        >>> cls = ff.getClasses()
        >>> len(cls)
        1
        >>> cls[0][0]
        'a'
        >>> ff._wtotal
        20.0
        """

        self[cls] = self.get(cls,0.0) + inputact
            
            
        self._wtotal += inputact

        # update by computing new proportions (ratios) and dropping weights whose
        # proportions are now less than wmin
        tot = self._wtotal
        for c in self.keys():   
            proport = (self[c]/tot)
            if proport < wmin:
                del(self[c])

    def getClasses(self, inputact=1.0, dmax=None):
        """
        @param inputact: [the input activation of this feature (self).]
        @param dmax: [the maximum number of categories we are interested in retrieving (all if None).]
        @return: [an association list of classes to weights in ranked (descending) order.]
        
        >>> f = FeatureFocus()
        >>> wmin = 0.1
        >>> f.update('a',2,wmin)
        >>> f.update('b',3,wmin)
        >>> cls = f.getClasses()
        >>> cls[0][0]
        'b'
        >>> '%0.2f' % cls[0][1]
        '0.30'
        >>> cls[1][0]
        'a'
        >>> '%0.2f' % cls[1][1]
        '0.20'
        >>> # Remember that there is downweighting.
        """        
        
        if len(self) <= dmax:
            dmax = None

        res = []
        tot =  float(self._wtotal)
        # downweight features seen < 10 times
        if tot < 10:
            tot = 10.0
        
        # rank classes by their proportional weights times the activation strength
        # of self (the feature).
        res = [(c, inputact * (self[c]/tot)) for c in self]
        res.sort(key=by_descending_count)
        if dmax is None:
            return res
        else:
            return res[:dmax]

def t(args):
    op = OptionParser()
    op.add_option("-p", "--profile", action="store_true", help="Whether or not to profile [default = False]", default=False)
    op.add_option("-f", "--inputf", type="string", \
                  help="The input file of 'class feat1 feat2 feat3 ...' lines\n"+\
                  "     To be used either for training or testing.")
    op.add_option("-m", "--modelf", type="string", \
                  help="The location of the (already-trained or to-be-trained) text model file.")
    op.add_option("-t", "--train", action="store_true", help="Whether or not to train [default = False]", default=False)
    op.add_option("-T", "--test", action="store_true", help="Whether or not to test [default = False]", default=False)
    op.add_option("-i", "--iterations", type="int", help="How many iterations to train (if training).", default=1)
    op.add_option("--dmax", type="int", help="The maximum number of classes each feature can predict [default = 25].", default=25)
    op.add_option("--wmin", type="float", help="The minimum proportion of the total focus feature group weight\n"+\
                  "that a class can be given by a feature [default = 0.01].", default=0.01)
    op.add_option("--dmarg", type="float", help="The margin by which the ff classifier must predict (one of) the\n"+\
                  "correct cat's without update being triggered [default = 0.5].", default=0.5)
    op.add_option("-b", "--boolean", action="store_true", help="Signals whether the context has Boolean (as opposed to real-valued)\n"+\
                  "activation in the input features.",default=False)
    op.add_option("-r", "--real_valued", action="store_true", help="Signals whether the context has real-valued (as opposed to Boolean)\n"+\
                  "activation in the input features.",\
                  default=False)    

    (options, args) = op.parse_args(args)
    if options.profile:
        prof = hotshot.Profile("classify.prof")
        prof.runcall(s,options,args)
        prof.close()
        stats = hotshot.stats.load("classify.prof")
        stats.strip_dirs()
        stats.sort_stats('time','calls')
        stats.print_stats(20)
    else:
        s(options,args)

def s(options,args):
    assert(not(options.real_valued and options.boolean) \
           and (options.real_valued or options.boolean))
        
    if options.test:        
        assert(not options.train)
        assert(options.inputf and os.path.exists(options.inputf))
        assert(options.modelf and os.path.exists(options.modelf))

    

        FFM = FeatureFocusModel()
        print >> sys.stderr, "Loading model %s ..." % options.modelf
        sys.stderr.flush()
        FFM.readFromFile(options.modelf)

        print >> sys.stderr, "Testing on data set %s: " % (options.inputf)
        sys.stderr.flush()
        
        # did the following to make sure can write file back out with no loss
        #         FFM.writeToFile("checkup.model")
        f = open(options.inputf,"r")


        
        # For IR-style stat's
        MRR_tot = 0.0
        R_1_tot = 0.0
        R_5_tot = 0.0        
        tot_correct = 0.0

        R_5_tot = 0.0        

        tot_correct = 0.0
        tot_instances = 0
        starttime = 0.0

        booleanFeats = None
        if options.real_valued:
            booleanFeats = False
        else:
            booleanFeats = True        
        try:
            starttime = time.time()
            for l in f:
                tot_instances += 1
                l = l.strip().split()

                ranking = FFM.predict(l, booleanFeats)
                MRR_tot += recipRank(l[0], ranking)
                tot_correct += recallAtK(l[0], ranking, 1)
                R_5_tot += recallAtK(l[0], ranking, 5)

                if tot_instances % 50000 == 0:
                    print >> sys.stderr, "running ave. at instance no. %d" % tot_instances, tot_correct/tot_instances,tot_instances
            endtime = time.time()

        finally:
            f.close()

        print "correct answer is top guess", tot_correct/tot_instances
        print "correct answer in top 5 guesses",R_5_tot/tot_instances
        print "MRR",MRR_tot/tot_instances
        print "total time in seconds (after loading model)", endtime - starttime
        print "total no. of instances", tot_instances
        return

    if not(options.train) and not(options.inputf and os.path.exists(options.inputf)):
        print >> sys.stderr, "Feature input file %s does not exist." % (options.inputf)
        return

    if options.train:
        FFM = FeatureFocusModel(options.dmax, options.dmarg, options.wmin, [])
                    
        starttime = time.time()

        for loop in range(options.iterations):

            print "Training on data set %s: iteration %s" % (options.inputf, str(loop+1))
            sys.stdout.flush()
                       
            f = open(options.inputf, 'r')

            # For IR-style stat's
            MRR_tot = 0.0
            R_1_tot = 0.0
            R_5_tot = 0.0        
            tot_instances = 0

            booleanFeats = None
            if options.real_valued:
                booleanFeats = False
            else:
                booleanFeats = True
                
            try:
                for l in f:
                    tot_instances += 1
                    l = l.strip().split()                    
                    ranking = FFM.learnFrom(l, booleanFeats)
                    MRR_tot += recipRank(l[0], ranking)
                    R_1_tot += recallAtK(l[0], ranking, 1)
                    R_5_tot += recallAtK(l[0], ranking, 5)
                
                message_center = ""
                if options.train:
                    message_center = "for iteration %s" % (str(loop+1))
                else:
                    # testing; talk of 'iterations' is meaningless.
                    message_center = "is"
                
                print "MRR "+message_center+": "+str(MRR_tot/tot_instances)
                print "R-1 recall "+message_center+": "+str(R_1_tot/tot_instances)
                print "R-5 recall "+message_center+": "+str(R_5_tot/tot_instances)            
            finally:
                f.close()

        
        FFM.writeToFile(options.modelf)

        endtime = time.time()
        print "total time in seconds %0.1d" % (endtime-starttime)
        print "ave. per-iteration time in seconds %0.1d" % ((endtime-starttime)/options.iterations)

if __name__=="__main__":
    doctest.testmod()



