package housing;

/**************************************************************************************************
 * Class to represent the mortgage policy regulator or Central Bank. It reads a number of policy
 * thresholds from the config object into local variables with the purpose of allowing for dynamic
 * policies varying those thresholds over time.
 *
 * @author daniel, Adrian Carro
 *
 *************************************************************************************************/

public class CentralBank {

    //------------------//
    //----- Fields -----//
    //------------------//

    // General fields
    private Config      config = Model.config;      // Passes the Model's configuration parameters object to a private field

    // Monetary policy
    private double      baseRate;

    // General soft policies thresholds
    private int monthsToCheckSoftLimits;            // Months to check for moving average of fraction of mortgages over their soft limit

    // LTV internal policy thresholds
    private double      firstTimeBuyerSoftMaxLTV;               // Loan-To-Value soft maximum for first-time buyer mortgages
    private double      firstTimeBuyerMaxFracOverSoftMaxLTV;    // Maximum fraction of first-time buyer mortgages allowed to exceed their LTV soft maximum
    private double      homeMoverSoftMaxLTV;                    // Loan-To-Value soft maximum for home mover mortgages
    private double      homeMoverMaxFracOverSoftMaxLTV;         // Maximum fraction of first-time buyer mortgages allowed to exceed their LTV soft maximum
    private double      buyToLetSoftMaxLTV;                     // Loan-To-Value soft maximum for buy-to-let mortgages
    private double      buyToLetMaxFracOverSoftMaxLTV;          // Maximum fraction of first-time buyer mortgages allowed to exceed their LTV soft maximum

    // LTI policy thresholds
    private double      firstTimeBuyerSoftMaxLTI;               // Loan-To-Income soft maximum for first-time buyer mortgages
    private double      firstTimeBuyerMaxFracOverSoftMaxLTI;    // Maximum fraction of first-time buyer mortgages allowed to exceed their LTI soft maximum
    private double      homeMoverSoftMaxLTI;                    // Loan-To-Income soft maximum for home mover mortgages
    private double      homeMoverMaxFracOverSoftMaxLTI;         // Maximum fraction of home mover mortgages allowed to exceed their LTI soft maximum
    private double      buyToLetSoftMaxLTI;                     // Loan-To-Income soft maximum for buy-to-let mortgages
    private double      buyToLetMaxFracOverSoftMaxLTI;          // Maximum fraction of buy-to-let mortgages allowed to exceed their LTI soft maximum

    // Affordability policy thresholds
    private double      hardMaxAffordability;       // Affordability hard maximum (monthly mortgage payment / household's monthly net employment income)

    // ICR policy thresholds
    private double      hardMinICR;                 // ICR hard minimum for the ratio of expected rental yield over interest monthly payment

    //-------------------//
    //----- Methods -----//
    //-------------------//

    void init() {
        // Set initial monetary policy
        baseRate = config.CENTRAL_BANK_INITIAL_BASE_RATE;
        // Set initial LTV mandatory policy thresholds
        firstTimeBuyerSoftMaxLTV = 0.9999; // TODO: Set these as non-binding initial parameters in config file
        firstTimeBuyerMaxFracOverSoftMaxLTV = config.CENTRAL_BANK_LTV_MAX_FRAC_OVER_SOFT_MAX_FTB;
        homeMoverSoftMaxLTV = 0.9999; // TODO: Set these as non-binding initial parameters in config file
        homeMoverMaxFracOverSoftMaxLTV = config.CENTRAL_BANK_LTV_MAX_FRAC_OVER_SOFT_MAX_HM;
        buyToLetSoftMaxLTV = 0.9999; // TODO: Set these as non-binding initial parameters in config file
        buyToLetMaxFracOverSoftMaxLTV = config.CENTRAL_BANK_LTV_MAX_FRAC_OVER_SOFT_MAX_BTL;
        // Set initial LTI mandatory policy thresholds
        firstTimeBuyerSoftMaxLTI = 15.0; // TODO: Set these as non-binding initial parameters in config file
        firstTimeBuyerMaxFracOverSoftMaxLTI = config.CENTRAL_BANK_LTI_MAX_FRAC_OVER_SOFT_MAX_FTB;
        homeMoverSoftMaxLTI = 15.0; // TODO: Set these as non-binding initial parameters in config file
        homeMoverMaxFracOverSoftMaxLTI = config.CENTRAL_BANK_LTI_MAX_FRAC_OVER_SOFT_MAX_HM;
        buyToLetSoftMaxLTI = 15.0; // TODO: Set these as non-binding initial parameters in config file
        buyToLetMaxFracOverSoftMaxLTI = config.CENTRAL_BANK_LTI_MAX_FRAC_OVER_SOFT_MAX_BTL;
        monthsToCheckSoftLimits = config.CENTRAL_BANK_MONTHS_TO_CHECK_SOFT_LIMITS;
        // Set initial affordability mandatory policy thresholds
        hardMaxAffordability = 0.9999; // TODO: Set these as non-binding initial parameters in config file
        // Set initial ICR mandatory policy thresholds
        hardMinICR = config.CENTRAL_BANK_ICR_HARD_MIN;
    }

    /**
     * This method implements the policy strategy of the Central Bank
     *
     * @param coreIndicators The current value of the core indicators
     * @param time The current model time
     */
    public void step(collectors.CoreIndicators coreIndicators, int time) {

        if (time >= config.CENTRAL_BANK_POLICY_APPLICATION_TIME) {
            // Update LTV mandatory policy thresholds
            firstTimeBuyerSoftMaxLTV = config.CENTRAL_BANK_LTV_SOFT_MAX_FTB;
            homeMoverSoftMaxLTV = config.CENTRAL_BANK_LTV_SOFT_MAX_HM;
            buyToLetSoftMaxLTV = config.CENTRAL_BANK_LTV_SOFT_MAX_BTL;
            // Update LTI mandatory policy thresholds
            firstTimeBuyerSoftMaxLTI = config.CENTRAL_BANK_LTI_SOFT_MAX_FTB;
            homeMoverSoftMaxLTI = config.CENTRAL_BANK_LTI_SOFT_MAX_HM;
            buyToLetSoftMaxLTI = config.CENTRAL_BANK_LTI_SOFT_MAX_BTL;
            // Update affordability mandatory policy thresholds
            hardMaxAffordability = config.CENTRAL_BANK_AFFORDABILITY_HARD_MAX;
        }

        /* Use this method to express the policy strategy of the central bank by setting the value of the various limits
         in response to the current value of the core indicators.

         Example policy: if house price growth is greater than 0.001 then FTB LTV limit is 0.75 otherwise (if house
         price growth is less than or equal to  0.001) FTB LTV limit is 0.95
         Example code:
            if(coreIndicators.getHousePriceGrowth() > 0.001) {
                firstTimeBuyerLTVLimit = 0.75;
            } else {
                firstTimeBuyerLTVLimit = 0.95;
            }
         */
    }


    //----- Getter/setter methods -----//

    double getFirstTimeBuyerSoftMaxLTV() { return firstTimeBuyerSoftMaxLTV; }

    double getHomeMoverSoftMaxLTV() { return homeMoverSoftMaxLTV; }

    double getBuyToLetSoftMaxLTV() { return buyToLetSoftMaxLTV; }

    double getFirstTimeBuyerMaxFracOverSoftMaxLTV() { return firstTimeBuyerMaxFracOverSoftMaxLTV; }

    double getHomeMoverMaxFracOverSoftMaxLTV() { return homeMoverMaxFracOverSoftMaxLTV; }

    double getBuyToLetMaxFracOverSoftMaxLTV() { return buyToLetMaxFracOverSoftMaxLTV; }

    double getFirstTimeBuyerSoftMaxLTI() { return firstTimeBuyerSoftMaxLTI; }

    double getHomeMoverSoftMaxLTI() { return homeMoverSoftMaxLTI; }

    double getBuyToLetSoftMaxLTI() { return buyToLetSoftMaxLTI; }

    double getFirstTimeBuyerMaxFracOverSoftMaxLTI() { return firstTimeBuyerMaxFracOverSoftMaxLTI; }

    double getHomeMoverMaxFracOverSoftMaxLTI() { return homeMoverMaxFracOverSoftMaxLTI; }

    double getBuyToLetMaxFracOverSoftMaxLTI() { return buyToLetMaxFracOverSoftMaxLTI; }

    int getMonthsToCheckSoftLimits() { return monthsToCheckSoftLimits; }

    double getHardMaxAffordability() { return hardMaxAffordability; }

    double getHardMinICR() { return hardMinICR; }

    double getBaseRate() { return baseRate; }

}
