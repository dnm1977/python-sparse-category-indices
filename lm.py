"""
A simple (toy) discrimiative language model.  In --train mode, it takes a text file, from stdin (utf-8, no <s> or </s>
wrapping the sentences) and extracts features based on the surrounding word contexts. These features are
written to stdout (utf-8).

In --test mode, it invokes ffclassifier.py (which is assumed to be in this same directory) to fill in blanks
in a test file.

E.g.,

Given the following sentences (utf-8, from stdin):

Bread and _
Chocolate and _
War and _

It will print out:

Bread and {prediction1}
Chocolate and {prediction2}
War and {prediction3}

The input file must have single underscores for each blank, and the output predictions will
be enclosed in curly brackets {...}.

This is not really a useful program, it just demonstrates how you might use ffclassifier.py.
"""
import sys, codecs, optparse, os, ffclassifier

op = optparse.OptionParser()
op.add_option("--train", action="store_true", help="whether to train", default=None)
op.add_option("--test", action="store_true", help="whether to test", default=None)
op.add_option("--model", type="string", help="model to use for testing", default=None)
op.add_option("--context", type="int", default=3, help="size of word context")

(ops,args) = op.parse_args()

try:
    assert( (ops.train or ops.test) and not(ops.train and ops.test) and not(ops.test and ops.model is None) )
except AssertionError, ae:
    print >> sys.stderr, "please specify exactly one of {--test, --train}"
    sys.exit(-1)

def fex(sent, i, N):
    """
    Extract features from context (size = N) for word at index 'i' (zero-based).
    """
    prior_words = sent[max((i-N),0):i]
    next_words = sent[i+1:(i+N+1)]
    prior_context = "prior="+("_".join(prior_words))
    following_context = "following="+("_".join(next_words))
    whole_context = "context="+(("_".join(prior_words) + "^" + ("_".join(next_words))))
    
    return (prior_context, following_context, whole_context)


(_,_,r,w) = codecs.lookup("utf-8")
sys.stdout = w(sys.stdout)
sys.stdin = r(sys.stdin)

ffmod = None
if ops.test:
    ffmod = ffclassifier.FeatureFocusModel()
    ffmod.readFromFile(ops.model)
    
for l in sys.stdin:
    sentence = l.strip().split()
    if sentence == []:
        continue
    sentence = ["<s>"] + sentence + ["</s>"]
    if ops.train:
        for j in range(1,len(sentence)-1):
            feats = fex(sentence, j, ops.context)
            wd = sentence[j]
            sys.stdout.write(wd + " " + " ".join(feats) + os.linesep)
    elif ops.test:
        indices_to_predict = [i for i in range(1,len(sentence)-1) if sentence[i] == "_"]
        predictions = []
        for j in indices_to_predict:
            fts = fex(sentence, j, ops.context)
            try:
                prediction = "{"+ffmod.rankedRetrieval(ffmod.makeContext(fts))[0][0]+"}"
            except Exception, e:
                prediction = "{NONE}"
            predictions.append(prediction)
        for i in range(len(sentence)):
            item = sentence[i]
            if item == "_":
                sentence[i] = predictions.pop(0)
        sys.stdout.write(" ".join(sentence[1:-1]) + os.linesep)
            
            

