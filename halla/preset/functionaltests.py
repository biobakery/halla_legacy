"""
Script containing all the functional tests in batch mode 

"""

def pr1():
		
		pOutHash = H.run_pr_test()
		csvw = csv.writer( sys.stdout , csv.excel_tab )
		astrHeaders = ["Var1", "Var2", "MID", "pPerm", "pPearson", "rPearson"]

		#Write the header
		csvw.writerow( astrHeaders )

		for k,v in pOutHash.items():
			iX, iY = k 
			csvw.writerow( [Name1[iX], Name2[iY]] + [v[j] for j in astrHeaders[2:]] )

		sys.stderr.write("Done!\n")
		#sys.stderr.write( str( pOutHash ) ) 

def rev1():
	H.run_rev1_test() 

def cake1():
	H.run_caketest()