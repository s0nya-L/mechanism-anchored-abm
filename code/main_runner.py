from simulation_core import *
import pandas as pd
import os

for m in range(20):
    for i in range(3):
        # Run for groups 0, 1, 2
        group = i

        filenameR = "data/finalResults/1simRESULTSgrp"+str(group)+".csv"
        filenameS = "data/simulationState/1simSTOCKgrp"+str(group)+".csv"
        filenameP = "data/simulationState/1simPURCHASINGgrp"+str(group)+".csv"
        filenameN = "data/rawNegotiations/1simNEGOTIATIONSgrp"+str(group)+".csv"
        offerLog = [] 
        negotiation_raw = [] 
        stock = []
        purchasing = []

        suppliers = []
        retailers = []
        s1 = supplier("Supplier1", group, suppliers)
        s2 = supplier("Supplier2", group, suppliers)
        s3 = supplier("Supplier3", group, suppliers)

        r1 = retailer("Retailer1", group, retailers)
        r2 = retailer("Retailer2", group, retailers)
        r3 = retailer("Retailer3", group, retailers)
        r4 = retailer("Retailer4", group, retailers)
        r5 = retailer("Retailer5", group, retailers)
        r6 = retailer("Retailer6", group, retailers)
        r7 = retailer("Retailer7", group, retailers)
        r8 = retailer("Retailer8", group, retailers)
        r9 = retailer("Retailer9", group, retailers)
        r10 = retailer("Retailer10", group, retailers)

        print(f"Beginning round {m} of group {group}!!")

        run_negotiations(rounds, offerLog, negotiation_raw, stock, purchasing, retailers, suppliers)
        print(f"Offer log for group {group}: {offerLog}")

        df = pd.DataFrame(offerLog)
        header = not os.path.exists(filenameR)
        df.to_csv(filenameR, mode='a', header=header, index=False)

        df1 = pd.DataFrame(stock)
        header = not os.path.exists(filenameS)
        df1.to_csv(filenameS, mode='a', header=header, index=False)

        df_neg = pd.DataFrame(negotiation_raw)
        header = not os.path.exists(filenameN)
        df_neg.to_csv(filenameN, mode='a', header=header, index=False)

        df_neg = pd.DataFrame(purchasing)
        header = not os.path.exists(filenameP)
        df_neg.to_csv(filenameP, mode='a', header=header, index=False)
