import halla, strudel 
import numpy 
from numpy import array 

######NEW PRESETS 
	# permutation_test_by_representative  -> norm_mi 
	# permutation_test_by_kpca_norm_mi -> 
	# permutation_test_by_kpca_pearson
	# permutation_test_by_cca_pearson 
	# permutation_test_by_cca_norm_mi 
s = strudel.Strudel( )

aQ = numpy.linspace(0.1,0.5,6)
astrSpike = ["vee", "sine"]
#astrSpike = ["parabola", "sine"]
#astrPreset = ["default", "kpca_norm_mi", "kpca_pearson", "cca_pearson", "cca_norm_mi"]
astrPreset = ["default", "kpca_norm_mi", "cca_pearson"] 
#astrPreset = ["kpca_pearson", "cca_pearson", "cca_norm_mi"]

#for q in aQ:

for strSpike in astrSpike:
	print "strSpike", strSpike 
	X = s.randmat( shape = (10,25) )
	Y,A = s.spike( X, strMethod = strSpike )

	# alla 
	print "running alla ..."
	P_alla = s.association( X,Y, strMethod = "norm_mi", bPval = 1 ).flatten()
	A_alla = 1.0 - P_alla 
	s.roc( A.flatten(), A_alla, strFile = "ALLA_roc_" + "_" + strSpike + ".pdf"  )
	s.alpha_fpr( A.flatten(), A_alla, strFile = "ALLA_alpha_" + "_" + strSpike + ".pdf" )

	for strPreset in astrPreset:
		print "strPreset", strPreset 

		# halla
		print "running halla ..."
		H  = halla.HAllA( X,Y )
		H.set_preset( strPreset )
		#H.set_q( q )
		aOut = H.run()
		P_halla = aOut[0].flatten()
		A_halla = 1.0 - P_halla
		s.roc( A.flatten(), A_halla, strFile = "HALLA_roc_" + strPreset + "_" + strSpike + ".pdf" )
		s.alpha_fpr( A.flatten(), A_halla, strFile = "HALLA_alpha_" + strPreset + "_" + strSpike + ".pdf" )

