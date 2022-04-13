package utilities;

import org.apache.commons.math3.random.MersenneTwister;

/**************************************************************************************************
 * Class to represent an arbitrarily shaped 1-dimensional Probability Density Function, with a
 * DoubleUnaryOperator class that returns the probability density for a given value.
 *
 * @author daniel, Adrian Carro
 *
 *************************************************************************************************/

public class Pdf {

    //------------------//
    //----- Fields -----//
    //------------------//

    private DoubleUnaryOperator     pdf;        // function that gives the pdf
    public double                   start;      // lowest value of x that has a non-zero probability
    public double                   end;        // highest value of x that has a non-zero probability
    private double []               inverseCDF; // pre-computed equi-spaced points on the inverse CDF including 0 and 1
    private int                     nSamples;   // number of sample points on the CDF
    private static final int        DEFAULT_CDF_SAMPLES = 100000;

    //------------------------//
    //----- Constructors -----//
    //------------------------//

    /**
     * Read the pdf from a binned .csv file. The format should be as specified in BinnedDataDouble.
     */
    public Pdf(String filename) {
        BinnedDataDouble data = new BinnedDataDouble(filename);
        setPdf(data);
    }

    public Pdf(final BinnedDataDouble data) { setPdf(data); }

    //-------------------//
    //----- Methods -----//
    //-------------------//

    public void setPdf(final BinnedDataDouble data) {
        pdf = new DoubleUnaryOperator() {
            public double applyAsDouble(double operand) {
                return data.getBinAt(operand) / data.getBinWidth();
            }
        };
        start = getStart(data);
        end = getEnd(data);
        nSamples = DEFAULT_CDF_SAMPLES;
        initInverseCDF();
    }

    private double getStart(final BinnedDataDouble data) {
        // Find the first bin with nonzero probability and store its left edge as start
        int firstNonZeroBin = 0;
        while ((firstNonZeroBin < data.size() - 1) && (data.get(firstNonZeroBin) == 0.0)) {
            firstNonZeroBin++;
        }
        if ((firstNonZeroBin == data.size() - 1) && (data.get(firstNonZeroBin) == 0.0)) {
            System.out.println("Error, trying to create a pdf but all bins have zero probability");
            System.exit(0);
        }
        return data.getSupportLowerBound() + firstNonZeroBin * data.getBinWidth();
    }

    private  double getEnd(final BinnedDataDouble data) {
        // Find the last bin with nonzero probability and store its right edge as end
        int lastNonZeroBin = data.size() - 1;
        while ((lastNonZeroBin >= 0) && (data.get(lastNonZeroBin) == 0.0)) {
            lastNonZeroBin--;
        }
        if ((lastNonZeroBin == 0) && (data.get(lastNonZeroBin) == 0.0)) {
            System.out.println("Error, trying to create a pdf but all bins have zero probability");
            System.exit(0);
        }
        return data.getSupportLowerBound() + (lastNonZeroBin + 1) * data.getBinWidth();
    }

    /**
     * Get probability density P(x)
     */
    public double density(double x) {
        if(x < start || x >= end) return(0.0);
        return(pdf.applyAsDouble(x));
    }

    public double inverseCumulativeProbability(double p) {
        if(p < 0.0 || p >= 1.0) throw(new IllegalArgumentException("p must be in the interval [0,1)"));
        int i = (int)(p * (nSamples - 1));
        double remainder = p * (nSamples - 1) - i;
        return((1.0 - remainder) * inverseCDF[i] + remainder * inverseCDF[i + 1]);
    }


    /**
     * Integrates the pdf over INTEGRATION_STEPS steps, starting at start + dx/2 and going up to end - dx/2, recording
     * the values of x at which the cumulative probability hits quantiles
     */
    private void initInverseCDF() {
        int INTEGRATION_STEPS = 262144;
        double cp;          // cumulative proability
        double targetcp;    // target cumulative probability
        double x;           // x in P(x)
        double dx;          // dx between samples
        double dcp_dx;
        int i;

        inverseCDF = new double[nSamples];
        dx = (end - start) / INTEGRATION_STEPS;
        x = start + dx /2.0;
        cp = 0.0;
        dcp_dx = 0.0;
        inverseCDF[0] = start;
        inverseCDF[nSamples-1] = end;
        for(i = 1; i < (nSamples - 1); ++i) {
            targetcp = i / (nSamples - 1.0);
            while(cp < targetcp && x < end) {
                dcp_dx = density(x);
                cp += dcp_dx* dx;
                x += dx;
            }
            if(x < end) {
                x += (targetcp - cp) / dcp_dx;
                cp = targetcp;
            } else {
                x = end;
            }
            inverseCDF[i] = x;
        }
    }

    /**
     * Obtain a random sample from the pdf
     */
    public double nextDouble(MersenneTwister rand) {
        return inverseCumulativeProbability(rand.nextDouble());
    }
}
