package collectors;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import java.util.Locale;

public class MicroDataRecorder {

    //------------------//
    //----- Fields -----//
    //------------------//

    private String          outputFolder;

    private PrintWriter     outfileHouseholdID;
    private PrintWriter     outfileEmploymentIncome;
    private PrintWriter     outfileRentalIncome;
    private PrintWriter     outfileBankBalance;
    private PrintWriter     outfileHousingNetWealth;
    private PrintWriter     outfileTotalDebt;
    private PrintWriter     outfileNHousesOwned;
    private PrintWriter     outfileHousingStatus;
    private PrintWriter     outfileAge;
    private PrintWriter     outfileConsumption;
    private PrintWriter     outfileSavingRate;

    private int             timeToStartMicroPrinting = 996;
    private int             freqOfMicroPrinting = 12;

    //------------------------//
    //----- Constructors -----//
    //------------------------//

    public MicroDataRecorder(String outputFolder) { this.outputFolder = outputFolder; }

    //-------------------//
    //----- Methods -----//
    //-------------------//

    public void openSingleRunSingleVariableFiles(int nRun, boolean recordHouseholdID, boolean recordEmploymentIncome,
                                                 boolean recordRentalIncome, boolean recordBankBalance,
                                                 boolean recordHousingNetWealth, boolean recordTotalDebt,
                                                 boolean recordNHousesOwned, boolean recordHousingStatus,
                                                 boolean recordAge, boolean recordConsumption,
                                                 boolean recordSavingRate) {
        if (recordHouseholdID) {
            try {
                outfileHouseholdID = new PrintWriter(outputFolder + "HouseholdID-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordEmploymentIncome) {
            try {
                outfileEmploymentIncome = new PrintWriter(outputFolder + "MonthlyGrossEmploymentIncome-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordRentalIncome) {
            try {
                outfileRentalIncome = new PrintWriter(outputFolder + "MonthlyGrossRentalIncome-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordBankBalance) {
            try {
                outfileBankBalance = new PrintWriter(outputFolder + "BankBalance-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordHousingNetWealth) {
            try {
                outfileHousingNetWealth = new PrintWriter(outputFolder + "HousingNetWealth-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordTotalDebt) {
            try {
                outfileTotalDebt = new PrintWriter(outputFolder + "TotalDebt-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordNHousesOwned) {
            try {
                outfileNHousesOwned = new PrintWriter(outputFolder + "NHousesOwned-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordHousingStatus) {
            try {
                outfileHousingStatus = new PrintWriter(outputFolder + "HousingStatus-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordAge) {
            try {
                outfileAge = new PrintWriter(outputFolder + "Age-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordConsumption) {
            try {
                outfileConsumption = new PrintWriter(outputFolder + "NonHousingConsumption-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        if (recordSavingRate) {
            try {
                outfileSavingRate = new PrintWriter(outputFolder + "SavingRate-run" + nRun
                        + ".csv", "UTF-8");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    }

    void timeStampSingleRunSingleVariableFiles(int time, boolean recordHouseholdID, boolean recordEmploymentIncome,
                                               boolean recordRentalIncome, boolean recordBankBalance,
                                               boolean recordHousingNetWealth, boolean recordTotalDebt,
                                               boolean recordNHousesOwned, boolean recordHousingStatus,
                                               boolean recordAge, boolean recordConsumption,
                                               boolean recordSavingRate) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            if (recordHouseholdID) {
                if (time != timeToStartMicroPrinting) {
                    outfileHouseholdID.println("");
                }
                outfileHouseholdID.print(time);
            }
            if (recordEmploymentIncome) {
                if (time != timeToStartMicroPrinting) {
                    outfileEmploymentIncome.println("");
                }
                outfileEmploymentIncome.print(time);
            }
            if (recordRentalIncome) {
                if (time != timeToStartMicroPrinting) {
                    outfileRentalIncome.println("");
                }
                outfileRentalIncome.print(time);
            }
            if (recordBankBalance) {
                if (time != timeToStartMicroPrinting) {
                    outfileBankBalance.println("");
                }
                outfileBankBalance.print(time);
            }
            if (recordHousingNetWealth) {
                if (time != timeToStartMicroPrinting) {
                    outfileHousingNetWealth.println("");
                }
                outfileHousingNetWealth.print(time);
            }
            if (recordTotalDebt) {
                if (time != timeToStartMicroPrinting) {
                    outfileTotalDebt.println("");
                }
                outfileTotalDebt.print(time);
            }
            if (recordNHousesOwned) {
                if (time != timeToStartMicroPrinting) {
                    outfileNHousesOwned.println("");
                }
                outfileNHousesOwned.print(time);
            }
            if (recordHousingStatus) {
                if (time != timeToStartMicroPrinting) {
                    outfileHousingStatus.println("");
                }
                outfileHousingStatus.print(time);
            }
            if (recordAge) {
                if (time != timeToStartMicroPrinting) {
                    outfileAge.println("");
                }
                outfileAge.print(time);
            }
            if (recordConsumption) {
                if (time != timeToStartMicroPrinting) {
                    outfileConsumption.println("");
                }
                outfileConsumption.print(time);
            }
            if (recordSavingRate) {
                if (time != timeToStartMicroPrinting) {
                    outfileSavingRate.println("");
                }
                outfileSavingRate.print(time);
            }
        }
    }

    void recordHouseholdID(int time, int householdID) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileHouseholdID.format(Locale.ROOT, "; %d", householdID);
        }
    }

    void recordEmploymentIncome(int time, double monthlyGrossEmploymentIncome) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileEmploymentIncome.format(Locale.ROOT, "; %.2f", monthlyGrossEmploymentIncome);
        }
    }

    void recordRentalIncome(int time, double monthlyGrossRentalIncome) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileRentalIncome.format(Locale.ROOT, "; %.2f", monthlyGrossRentalIncome);
        }
    }

    void recordBankBalance(int time, double bankBalance) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileBankBalance.format(Locale.ROOT, "; %.2f", bankBalance);
        }
    }

    void recordHousingNetWealth(int time, double housingNetWealth) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileHousingNetWealth.format(Locale.ROOT, "; %.2f", housingNetWealth);
        }
    }

    void recordTotalDebt(int time, double totalDebt) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileTotalDebt.format(Locale.ROOT, "; %.2f", totalDebt);
        }
    }

    void recordNHousesOwned(int time, int nHousesOwned) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileNHousesOwned.format(Locale.ROOT, "; %d", nHousesOwned);
        }
    }

    void recordHousingStatus(int time, int housingStatus) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileHousingStatus.format(Locale.ROOT, "; %d", housingStatus);
        }
    }

    void recordAge(int time, double age) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileAge.format(Locale.ROOT, "; %.2f", age);
        }
    }

    void recordConsumption(int time, double consumption) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileConsumption.format(Locale.ROOT, "; %.2f", consumption);
        }
    }

    void recordSavingRate(int time, double savingRate) {
        if (time % freqOfMicroPrinting == 0 && time >= timeToStartMicroPrinting) {
            outfileSavingRate.format(Locale.ROOT, "; %.4f", savingRate);
        }
    }

    public void finishRun(boolean recordHouseholdID, boolean recordEmploymentIncome, boolean recordRentalIncome,
                          boolean recordBankBalance, boolean recordHousingNetWealth, boolean recordTotalDebt,
                          boolean recordNHousesOwned, boolean recordHousingStatus, boolean recordAge,
                          boolean recordConsumption, boolean recordSavingRate) {
        if (recordHouseholdID) {
            outfileHouseholdID.close();
        }
        if (recordEmploymentIncome) {
            outfileEmploymentIncome.close();
        }
        if (recordRentalIncome) {
            outfileRentalIncome.close();
        }
        if (recordBankBalance) {
            outfileBankBalance.close();
        }
        if (recordHousingNetWealth) {
            outfileHousingNetWealth.close();
        }
        if (recordTotalDebt) {
            outfileTotalDebt.close();
        }
        if (recordNHousesOwned) {
            outfileNHousesOwned.close();
        }
        if (recordHousingStatus) {
            outfileHousingStatus.close();
        }
        if (recordAge) {
            outfileAge.close();
        }
        if (recordConsumption) {
            outfileConsumption.close();
        }
        if (recordSavingRate) {
            outfileSavingRate.close();
        }
    }
}
