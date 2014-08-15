import halla, strudel 
import numpy 
from numpy import array 


#halla.stats.bag2association 

######NEW PRESETS 
	# permutation_test_by_representative  -> norm_mi 
	# permutation_test_by_kpca_norm_mi -> 
	# permutation_test_by_kpca_pearson
	# permutation_test_by_cca_pearson 
	# permutation_test_by_cca_norm_mi 
s = strudel.Strudel( )

aQ = numpy.linspace(0.5,1.0,6)
astrSpike = ["vee"]
#astrSpike = ["parabola", "sine"]
#astrPreset = ["default", "kpca_norm_mi", "kpca_pearson", "cca_pearson", "cca_norm_mi"]
astrPreset = ["default", "kpca_norm_mi", "cca_pearson"] 
#astrPreset = ["kpca_pearson", "cca_pearson", "cca_norm_mi"]

for strSpike in astrSpike:

	X,Y,A,iX,iY = None, None, None, None, None 

	for q in aQ:
		
	
		if Y == None:
			print "strSpike", strSpike 
			X = s.randmat( shape = (10,100) )
			Y,A = s.spike( X, strMethod = strSpike, sparsity = 1.0 )

			iX = X.shape[0]
			iY = Y.shape[0]

		#print "q", q 

		A_alla = None 

		for strPreset in astrPreset:
			print "strPreset", strPreset 

			# alla 
			if A_alla == None:
				print "running alla ..."
				P_alla = s.association( X,Y, strMethod = "norm_mi", bPval = 1 ).flatten()
				A_alla = 1.0 - P_alla 

			# halla
			print "running halla ..."
			H  = halla.HAllA( X,Y )
			H.set_preset( strPreset )
			H.set_q( q )
			H.set_verbose( True )
			aOut = H.run()
			A_conditional_flattened, A_halla_conditional_flattened = halla.stats.bag2association(aOut[0], A)
			A_alla_conditional_flattened, _ =  halla.stats.bag2association(aOut[0], numpy.reshape( A_alla, (iX,iY) ) )

			print H.q 
			print aOut[0] 

			#print A_conditional_flattened
			#print A_halla_conditional_flattened
			#print A_alla_conditional_flattened

			### save the plots 

			if A_alla_conditional_flattened != []:
				s.roc( A_conditional_flattened, A_alla_conditional_flattened, strFile = "ALLA_roc_" + strSpike + ".pdf", strTitle = "ALLA_roc_" + strSpike  )
				s.alpha_fpr( A_conditional_flattened, A_alla_conditional_flattened, strFile = "ALLA_alpha_" + strSpike + ".pdf", strTitle = "ALLA_alpha_" + strSpike )

			if A_halla_conditional_flattened != []:
				s.roc( A_conditional_flattened, A_halla_conditional_flattened, strFile = "HALLA_roc_" + str(q) + "_" + strPreset + "_" + strSpike + ".pdf", strTitle = "HALLA_roc_" + str(q) + "_" + strPreset + "_" + strSpike )
				s.alpha_fpr( A_conditional_flattened, A_halla_conditional_flattened, strFile = "HALLA_alpha_" + str(q) + "_" + strPreset + "_" + strSpike + ".pdf", strTitle = "HALLA_alpha_" + str(q) + "_" + strPreset + "_" + strSpike )

