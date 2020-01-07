package housing;

import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

/**************************************************************************************************
 * Class to encapsulate all the configuration parameters of the model. It also contains all methods
 * needed to read these parameter values from a configuration properties file.
 *
 * @author Adrian Carro
 * @since 20/02/2017
 *
 *************************************************************************************************/
public class Config {

    //---------------------------------//
    //----- Fields and subclasses -----//
    //---------------------------------//

    /** Declaration of parameters **/

    // General model control parameters
    int SEED;                                           // Seed for the random number generator
    int N_STEPS;				                        // Simulation duration in time steps
    int N_SIMS; 					                    // Number of simulations to run (monte-carlo)
    public int TIME_TO_START_RECORDING_TRANSACTIONS;    // Time step to start recording transactions (to avoid too large files)
    public boolean recordTransactions;			        // True to write data for each transaction
    boolean recordNBidUpFrequency;			            // True to write the frequency of the number of bid-ups
    boolean recordCoreIndicators;		                // True to write time series for each core indicator
    boolean recordQualityBandPrice;                     // True to write time series of prices for each quality band to a single file per run
    public boolean recordEmploymentIncome;              // True to write individual household monthly gross employment income data
    public boolean recordRentalIncome;                  // True to write individual household monthly gross rental income data (after market clearing)
    public boolean recordBankBalance;                   // True to write individual household liquid wealth (bank balance) data (after market clearing)
    public boolean recordHousingWealth;                 // True to write individual household housing wealth data (after market clearing, assuming constant house prices!)
    public boolean recordNHousesOwned;                  // True to write individual household number of houses owned data (after market clearing)
    public boolean recordAge;                           // True to write individual household age of the household representative person
    public boolean recordSavingRate;                    // True to write individual household saving rate data [1 - (taxExpenses + housing expenses(except deposits) + essentialConsumption + nonEssentialConsumption)/monthlyGrossTotalIncome]

    // House parameters
    public int N_QUALITY;                   // Number of quality bands for houses

    // Housing market parameters
    private double DAYS_UNDER_OFFER;            // Time (in days) that a house remains under offer
    double BIDUP;                               // Smallest proportional increase in price that can cause a gazump
    public double MARKET_AVERAGE_PRICE_DECAY;   // Decay constant for the exponential moving average of sale prices
    public double HOUSE_PRICES_SCALE;           // Scale parameter for the log-normal distribution of house prices (logarithm of median house price = mean and median of logarithmic house prices)
    public double HOUSE_PRICES_SHAPE;           // Shape parameter for the log-normal distribution of house prices (standard deviation of logarithmic house prices)
    public double RENTAL_PRICES_SCALE;          // Scale parameter for the log-normal distribution of house rental prices (logarithm of median house rental price = mean and median of logarithmic house rental prices)
    public double RENTAL_PRICES_SHAPE;          // Shape parameter for the log-normal distribution of house rental prices (standard deviation of logarithmic house rental prices)
    public double RENT_GROSS_YIELD;             // Profit margin for buy-to-let investors

    // Demographic parameters
    public int TARGET_POPULATION;           // Target number of households
    public String DATA_AGE_DISTRIBUTION;    // Address for data on the age distribution of household representative persons

    // Household parameters
    public String DATA_INCOME_GIVEN_AGE;    // Address for conditional probability of total gross non-rent income given age
    public String DATA_WEALTH_GIVEN_INCOME; // Address for conditional probability of liquid wealth given total gross non-rent income

    // Household behaviour parameters: buy-to-let
    String DATA_BTL_PROBABILITY;            // Probability of being a buy-to-let investor per income percentile bin
    double BTL_PROBABILITY_MULTIPLIER;      // # Multiplier for the probability of being a buy-to-let investor
    double FUNDAMENTALIST_CAP_GAIN_COEFF;   // Weight that fundamentalists put on cap gain
    double TREND_CAP_GAIN_COEFF;			// Weight that trend-followers put on cap gain
    double P_FUNDAMENTALIST; 			    // Probability that a BTL investor is a fundamentalist versus a trend-follower
    // Household behaviour parameters: rent
    int TENANCY_LENGTH_MIN;                 // Rental contract lengths are drawn from a uniform (discrete) distribution between TENANCY_LENGTH_MIN and TENANCY_LENGTH_MAX
    int TENANCY_LENGTH_MAX;                 // Rental contract lengths are drawn from a uniform (discrete) distribution between TENANCY_LENGTH_MIN and TENANCY_LENGTH_MAX
    String DATA_RENT_BID_FRACTION;          // Proportion of income households bid on the rental market, per income bin
    double BID_RENT_AS_FRACTION_OF_INCOME;  // Proportion of income households bid on the rental market
    double PSYCHOLOGICAL_COST_OF_RENTING;   // Annual psychological cost of renting
    double SENSITIVITY_RENT_OR_PURCHASE;    // Sensitivity parameter of the decision between buying and renting
    // Household behaviour parameters: general
    double HPA_EXPECTATION_FACTOR;          // Dampening (or multiplier) factor for previous trend when computing future HPI growth expectations
    double HPA_EXPECTATION_CONST;           // Constant to be added or subtracted from previous trend when computing future HPI growth expectations
    public int HPA_YEARS_TO_CHECK;          // Number of years of the HPI record to check when computing the annual HPA
    // Household behaviour parameters: sale price reduction
    double P_SALE_PRICE_REDUCE;             // Monthly probability of reducing the price of a house on the market
    double REDUCTION_MU;                    // Mean percentage reduction for prices of houses on the market
    double REDUCTION_SIGMA;                 // Standard deviation of percentage reductions for prices of houses on the market
    // Household behaviour parameters: consumption
    double ESSENTIAL_CONSUMPTION_FRACTION;  // Fraction of Government support necessarily spent monthly by all households as essential consumption
    // Household behaviour parameters: initial sale price
    String DATA_INITIAL_SALE_MARKUP_DIST;   // Address for probability distribution of sale price mark-ups
    // Household behaviour parameters: initial rent price
    String DATA_INITIAL_RENT_MARKUP_DIST;   // Address for probability distribution of rent price mark-ups
    // Household behaviour parameters: buyer's desired expenditure
    double BUY_SCALE;                       // Scale, number of annual salaries (raised to the BUY_EXPONENT power) the buyer is willing to spend
    double BUY_EXPONENT;                    // Exponent to which the annual gross employment income of the household is raised when computing its budget
    double BUY_WEIGHT_HPA;                  // Weight given to house price appreciation when deciding how much to spend
    double BUY_MU;                          // Mean of the normal noise used to create a log-normal variate, which is then used as a multiplicative noise
    double BUY_SIGMA;                       // Standard deviation of the normal noise used to create a log-normal variate, which is then used as a multiplicative noise
    // Household behaviour parameters: rent reduction
    double RENT_REDUCTION;                  // Percentage reduction of demanded rent for every month the property is in the market, not rented
    // Household behaviour parameters: downpayment
    double DOWNPAYMENT_BANK_BALANCE_FOR_CASH_SALE;  // If bankBalance/housePrice is above this, payment will be made fully in cash
    double DOWNPAYMENT_FTB_SCALE;                   // Scale parameter for the log-normal distribution of downpayments by first-time-buyers
    double DOWNPAYMENT_FTB_SHAPE;                   // Shape parameter for the log-normal distribution of downpayments by first-time-buyers
    double DOWNPAYMENT_OO_SCALE;                    // Scale parameter for the log-normal distribution of downpayments by owner-occupiers
    double DOWNPAYMENT_OO_SHAPE;                    // Shape parameter for the log-normal distribution of downpayments by owner-occupiers
    double DOWNPAYMENT_MIN_INCOME;                  // Minimum income percentile to consider any downpayment, below this level, downpayment is set to 0
    double DOWNPAYMENT_BTL_MEAN;                    // Average down-payment, as a percentage of house price, for but-to-let investors
    double DOWNPAYMENT_BTL_EPSILON;                 // Standard deviation of the noise for down-payments by buy-to-let investors
    // Household behaviour parameters: selling decision
    private double HOLD_PERIOD;             // Average period, in years, for which owner-occupiers hold their houses
    // Household behaviour parameters: BTL buy/sell choice
    double BTL_CHOICE_INTENSITY;            // Shape parameter, or intensity of choice on effective yield
    double BTL_CHOICE_MIN_BANK_BALANCE;     // Minimun bank balance, as a percentage of the desired bank balance, to buy new properties

    // Bank parameters
    private int MORTGAGE_DURATION_YEARS;    // Mortgage duration in years
    double BANK_INITIAL_RATE;               // Private bank initial interest rate
    double BANK_CREDIT_SUPPLY_TARGET;       // Bank's target supply of credit per household per month
    double BANK_D_DEMAND_D_INTEREST;        // Rate of change of the demand for credit in response to a change in the interest rate (in pounds per point)
    double BANK_MAX_FTB_LTV;                // Maximum LTV ratio that the private bank would allow for first-time-buyers
    double BANK_MAX_OO_LTV;                 // Maximum LTV ratio that the private bank would allow for owner-occupiers
    double BANK_MAX_BTL_LTV;                // Maximum LTV ratio that the private bank would allow for BTL investors
    double BANK_MAX_FTB_LTI;                // Maximum LTI ratio that the private bank would allow for first-time-buyers (private bank's hard limit)
    double BANK_MAX_OO_LTI;                 // Maximum LTI ratio that the private bank would allow for owner-occupiers (private bank's hard limit)


    // Central bank parameters
    double CENTRAL_BANK_INITIAL_BASE_RATE;      // Central Bank initial base rate
    double CENTRAL_BANK_MAX_FTB_LTI;		    // Maximum LTI ratio that the bank would allow for first-time-buyers when not regulated
    double CENTRAL_BANK_MAX_OO_LTI;		        // Maximum LTI ratio that the bank would allow for owner-occupiers when not regulated
    double CENTRAL_BANK_FRACTION_OVER_MAX_LTI;  // Maximum fraction of mortgages that the bank can give over the LTI ratio limit
    double CENTRAL_BANK_AFFORDABILITY_COEFF;    // Maximum fraction of the household's income to be spent on mortgage repayments under stressed conditions
    double CENTRAL_BANK_BTL_STRESSED_INTEREST;  // Interest rate under stressed condition for BTL investors when calculating interest coverage ratios (ICR)
    double CENTRAL_BANK_MAX_ICR;                // Interest coverage ratio (ICR) limit imposed by the central bank

    // Construction parameters
    private int UK_HOUSEHOLDS;              // Number of households in the UK, used to compute core indicators and the ratio of houses per household
    private int UK_DWELLINGS;               // Number of dwellings in the UK, used to compute the ratio of houses per household

    // Government parameters
    double GOVERNMENT_GENERAL_PERSONAL_ALLOWANCE;           // General personal allowance to be deducted when computing taxable income
    double GOVERNMENT_INCOME_LIMIT_FOR_PERSONAL_ALLOWANCE;  // Limit of income above which personal allowance starts to decrease £1 for every £2 of income above this limit
    public double GOVERNMENT_MONTHLY_INCOME_SUPPORT;        // Income support for a couple, both over 18 years old (Jobseeker's allowance)
    public String DATA_TAX_RATES;                           // Address for tax bands and rates data
    public String DATA_NATIONAL_INSURANCE_RATES;            // Address for national insurance bands and rates data

    /** Construction of objects to contain derived parameters and constants **/

    // Create object containing all constants
    public Config.Constants constants = new Constants();

    // Finally, create object containing all derived parameters
    public Config.DerivedParams derivedParams = new DerivedParams();

    /**
     * Class to contain all parameters which are not read from the configuration (.properties) file, but derived,
     * instead, from these configuration parameters
     */
    public class DerivedParams {
        // Housing market parameters
        public int HPI_RECORD_LENGTH;   // Number of months to record HPI (to compute price growth at different time scales)
        double MONTHS_UNDER_OFFER;      // Time (in months) that a house remains under offer
        public double E;                // Decay constant for averaging months on market (in transactions)
        public double G;                // Decay constant for averageListPrice averaging (in transactions)
        double HOUSE_PRICES_MEAN;       // Mean of reference house prices (scale + shape**2/2)
        // Household behaviour parameters: general
        double MONTHLY_P_SELL;          // Monthly probability for owner-occupiers to sell their houses
        // Household behaviour parameters: rent
        public double TENANCY_LENGTH_AVERAGE;  // Average number of months a tenant will stay in a rented house
        // Bank parameters
        int N_PAYMENTS;                 // Number of monthly repayments (mortgage duration in months)
        // House rental market parameters
        public double K;                // Decay factor for exponential moving average of gross yield from rentals
        // Construction parameters
        double UK_HOUSES_PER_HOUSEHOLD; // Target ratio of houses per household

        public double getE() { return E; }

        public int getHPIRecordLength() { return HPI_RECORD_LENGTH; }

        public double getHousePricesMean() { return HOUSE_PRICES_MEAN; }

    }

    /**
     * Class to contain all constants (not read from the configuration file nor derived from it)
     */
    public class Constants {
        final int DAYS_IN_MONTH = 30;
        final public int MONTHS_IN_YEAR = 12;
    }

    //------------------------//
    //----- Constructors -----//
    //------------------------//

    /**
     * Empty constructor, mainly used for copying local instances of the Model Config instance into other classes
     */
    public Config () {}

    /**
     * Constructor with full initialization, used only for the original Model Config instance
     */
    public Config (String configFileName) { getConfigValues(configFileName); }

    //-------------------//
    //----- Methods -----//
    //-------------------//

    public int getUKHouseholds() { return UK_HOUSEHOLDS; }

    /**
     * Method to read configuration parameters from a configuration (.properties) file
     * @param   configFileName    String with name of configuration (.properties) file (address inside source folder)
     */
    private void getConfigValues(String configFileName) {
        // Try-with-resources statement
        try (FileReader fileReader = new FileReader(configFileName)) {
            Properties prop = new Properties();
            prop.load(fileReader);
            // Check that all parameters declared in the configuration (.properties) file are also declared in this class
            try {
                Set<String> setOfFields = new HashSet<>();
                for (Field field : this.getClass().getDeclaredFields()) {
                    setOfFields.add(field.getName());
                }
                for (String property: prop.stringPropertyNames()) {
                    if (!setOfFields.contains(property)) {
                        throw new UndeclaredPropertyException(property);
                    }
                }
            } catch (UndeclaredPropertyException upe) {
                upe.printStackTrace();
            }
            // Run through all the fields of the Class using reflection
            for (Field field : this.getClass().getDeclaredFields()) {
                try {
                    // For int fields, parse the int with appropriate exception handling
                    if (field.getType().toString().equals("int")) {
                        try {
                            if (prop.getProperty(field.getName()) == null) {throw new FieldNotInFileException(field);}
                            field.set(this, Integer.parseInt(prop.getProperty(field.getName())));
                        } catch (NumberFormatException nfe) {
                            System.out.println("Exception " + nfe + " while trying to parse the field " +
                                    field.getName() + " for an integer");
                            nfe.printStackTrace();
                        } catch (IllegalAccessException iae) {
                            System.out.println("Exception " + iae + " while trying to set the field " +
                                    field.getName());
                            iae.printStackTrace();
                        } catch (FieldNotInFileException fnife) {
                            fnife.printStackTrace();
                        }
                    // For double fields, parse the double with appropriate exception handling
                    } else if (field.getType().toString().equals("double")) {
                        try {
                            if (prop.getProperty(field.getName()) == null) {throw new FieldNotInFileException(field);}
                            field.set(this, Double.parseDouble(prop.getProperty(field.getName())));
                        } catch (NumberFormatException nfe) {
                            System.out.println("Exception " + nfe + " while trying to parse the field " +
                                    field.getName() + " for an double");
                            nfe.printStackTrace();
                        } catch (IllegalAccessException iae) {
                            System.out.println("Exception " + iae + " while trying to set the field " +
                                    field.getName());
                            iae.printStackTrace();
                        } catch (FieldNotInFileException fnife) {
                            fnife.printStackTrace();
                        }
                    // For boolean fields, parse the boolean with appropriate exception handling
                    } else if (field.getType().toString().equals("boolean")) {
                        try {
                            if (prop.getProperty(field.getName()) == null) {throw new FieldNotInFileException(field);}
                            if (prop.getProperty(field.getName()).equals("true") ||
                                    prop.getProperty(field.getName()).equals("false")) {
                                field.set(this, Boolean.parseBoolean(prop.getProperty(field.getName())));
                            } else {
                                throw new BooleanFormatException("For input string \"" +
                                        prop.getProperty(field.getName()) + "\"");
                            }
                        } catch (BooleanFormatException bfe) {
                            System.out.println("Exception " + bfe + " while trying to parse the field " +
                                    field.getName() + " for a boolean");
                            bfe.printStackTrace();
                        } catch (IllegalAccessException iae) {
                            System.out.println("Exception " + iae + " while trying to set the field " +
                                    field.getName());
                            iae.printStackTrace();
                        } catch (FieldNotInFileException fnife) {
                            fnife.printStackTrace();
                        }
                    // For string fields, parse the string with appropriate exception handling
                    } else if (field.getType().toString().equals("class java.lang.String")) {
                        try {
                            if (prop.getProperty(field.getName()) == null) {throw new FieldNotInFileException(field);}
                            field.set(this, prop.getProperty(field.getName()).replace("\"", "").replace("\'", ""));
                        } catch (IllegalAccessException iae) {
                            System.out.println("Exception " + iae + " while trying to set the field " +
                                    field.getName());
                            iae.printStackTrace();
                        } catch (FieldNotInFileException fnife) {
                            fnife.printStackTrace();
                        }
                    // For unrecognised field types, except derivedParams and constants, throw exception
                    } else if (!field.getName().equals("derivedParams") && !field.getName().equals("constants")) {
                        throw new UnrecognisedFieldTypeException(field);
                    }
                } catch (UnrecognisedFieldTypeException ufte) {
                    ufte.printStackTrace();
                }
            }
        } catch (IOException ioe) {
            System.out.println("Exception " + ioe + " while trying to read file '" + configFileName + "'");
            ioe.printStackTrace();
        }
        // Finally, compute and set values for all derived parameters
        setDerivedParams();
    }

    /**
     * Method to compute and set values for all derived parameters
     */
    private void setDerivedParams() {
        // Housing market parameters
        derivedParams.HPI_RECORD_LENGTH = HPA_YEARS_TO_CHECK*constants.MONTHS_IN_YEAR + 3;  // Plus three months in a quarter
        derivedParams.MONTHS_UNDER_OFFER = DAYS_UNDER_OFFER/constants.DAYS_IN_MONTH;
        derivedParams.E = Math.exp(-1.0 / (0.02 * TARGET_POPULATION));          // TODO: Clarify where does this 0.2 come from, provide explanation for this formula
        derivedParams.G = Math.exp(-N_QUALITY / (0.02 * TARGET_POPULATION));    // TODO: Clarify where does this 0.2 come from, provide explanation for this formula
        derivedParams.HOUSE_PRICES_MEAN = Math.exp(HOUSE_PRICES_SCALE + HOUSE_PRICES_SHAPE*HOUSE_PRICES_SHAPE/2.0); // Mean of a log-normal distribution
        // Household behaviour parameters: general
        derivedParams.MONTHLY_P_SELL = 1.0/(HOLD_PERIOD*constants.MONTHS_IN_YEAR);
        // Household behaviour parameters: rent
        derivedParams.TENANCY_LENGTH_AVERAGE = (TENANCY_LENGTH_MIN + TENANCY_LENGTH_MAX) / 2.0;
        // Bank parameters
        derivedParams.N_PAYMENTS = MORTGAGE_DURATION_YEARS*constants.MONTHS_IN_YEAR;
        // House rental market parameters
        derivedParams.K = Math.exp(-10000.0/(TARGET_POPULATION*50.0));  // TODO: Are these decay factors well-suited? Any explanation, reasoning behind the numbers chosen? Also, they're not reported in the paper!
        // Construction parameters
        derivedParams.UK_HOUSES_PER_HOUSEHOLD = (double)UK_DWELLINGS / UK_HOUSEHOLDS; // Target ratio of houses per household
    }

    /**
     * Equivalent to NumberFormatException for detecting problems when parsing for boolean values
     */
    public class BooleanFormatException extends RuntimeException {
        BooleanFormatException(String message) { super(message); }
    }

    /**
     * Exception for detecting unrecognised (not implemented) field types
     */
    public class UnrecognisedFieldTypeException extends Exception {
        UnrecognisedFieldTypeException(Field field) {
            super("Field type \"" + field.getType().toString() + "\", found at field \"" + field.getName() +
                    "\", could not be recognised as any of the implemented types");
        }
    }

    /**
     * Exception for detecting fields declared in the code but not present in the config.properties file
     */
    public class FieldNotInFileException extends Exception {
        FieldNotInFileException(Field field) {
            super("Field \"" + field.getName() + "\" could not be found in the config.properties file");
        }
    }

    /**
     * Exception for detecting properties present in the config.properties file but not declared in the code
     */
    public class UndeclaredPropertyException extends Exception {
        UndeclaredPropertyException(Object property) {
            super("Property \"" + property + "\" could not be found among the fields declared within the Config class");
        }
    }
}
