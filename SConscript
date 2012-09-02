import os
import sfle
import sys

Import( "*" )
pE = DefaultEnvironment( )

c_strMetadatum				= "study_day2"

c_fileInputMetaphlanPCL		= sfle.d( pE, fileDirInput, "HMP.ab.txt" )
c_fileInputModuleP			= sfle.d( pE, c_fileDirOutput, "kegg", "modulep" )
c_fileInputOTUsPCL			= sfle.d( pE, fileDirInput, "Saliva.pcl.gz" )
c_fileInputIMGPCL			= sfle.d( pE, fileDirInput, "img_kos.pcl.gz" )
c_fileInputMapKEGGTXT		= sfle.d( pE, fileDirInput, "map_kegg.txt" )
c_fileInputMetadataTXT		= sfle.d( pE, fileDirInput, "pds_metadata.txt" )
c_fileInputExcludeDAT		= sfle.d( pE, fileDirInput, "hmp_exclude.dat" )
c_fileInputKOsPCL			= sfle.d( pE, fileDirInput, "kegg_kos.pcl.gz" )
c_fileInputPathogensTXT		= sfle.d( pE, c_fileDirOutput, "hmp_analysis", "pathogens.txt" )
#c_fileInputPathogensTXT		= sfle.d( pE, fileDirInput, "pathogens.txt" )

c_fileTDPCL					= sfle.d( pE, fileDirTmp, "TD.pcl" )
c_fileTDTSV					= File( str(c_fileTDPCL).replace( sfle.c_strSufPCL, sfle.c_strSufTSV ) )
c_fileTDPreTSV				= File( str(c_fileTDPCL).replace( sfle.c_strSufPCL, "_pre" + sfle.c_strSufTSV ) )
c_fileTDKOsPCL				= sfle.d( pE, fileDirTmp, "TDKOs.pcl" )
c_fileTDKOsTSV				= File( str(c_fileTDKOsPCL).replace( sfle.c_strSufPCL, sfle.c_strSufTSV ) )
c_fileTDKOsPreTSV			= File( str(c_fileTDKOsPCL).replace( sfle.c_strSufPCL, "_pre" + sfle.c_strSufTSV ) )
c_fileMetaphlanPCL			= sfle.d( pE, fileDirTmp, "metaphlan.pcl" )
c_fileMetaphlanPrePCL		= File( str(c_fileMetaphlanPCL).replace( sfle.c_strSufPCL, "_pre" + sfle.c_strSufPCL ) )
c_fileMetaphlanTSV			= File( str(c_fileMetaphlanPCL).replace( sfle.c_strSufPCL, sfle.c_strSufTSV ) )
c_fileKOsPCL				= sfle.d( pE, fileDirTmp, sfle.rebase( c_fileInputKOsPCL, sfle.c_strSufGZ, "" ) )
c_fileKOsPrePCL				= File( str(c_fileKOsPCL).replace( sfle.c_strSufPCL, "_pre" + sfle.c_strSufPCL ) )
c_fileKOsTSV				= File( str(c_fileKOsPCL).replace( sfle.c_strSufPCL, sfle.c_strSufTSV ) )
c_fileCorrelationsTXT		= sfle.d( pE, fileDirTmp, "correlations.txt" )
c_fileMetaphlanPDF			= sfle.d( pE, fileDirOutput, "metaphlan.pdf" )
c_fileOTUsPDF				= sfle.d( pE, fileDirOutput, "otus.pdf" )
c_fileCarriagePDF			= sfle.d( pE, fileDirOutput, "carriage.pdf" )
c_fileModulesPCL			= sfle.d( pE, fileDirOutput, "modules.pcl" )
c_fileBugsTXT				= sfle.d( pE, fileDirOutput, "bugs.txt" )
c_fileModulesPDF			= sfle.d( pE, fileDirOutput, "modules.pdf" )
c_fileCorrelationsPDF		= sfle.d( pE, fileDirOutput, "correlations.pdf" )

c_fileProgPCL2Img			= sfle.d( pE, fileDirSrc, "pcl2img.py" )
c_fileProgPCL2CDF			= sfle.d( pE, fileDirSrc, "pcl2cdf.py" )
c_fileProgPathcov			= sfle.d( pE, fileDirSrc, "pathcov.py" )
c_fileProgModules2Heatmap	= sfle.d( pE, fileDirSrc, "modules2heatmap.py" )
c_fileProgMetaphlan2PCL		= sfle.d( pE, "../hmp_analysis", sfle.c_strDirSrc, "metaphlan2pcl.py" )
c_fileProgMergeMetadata		= sfle.find( c_fileDirInput, "merge_metadata.py", pE )
c_fileProgCorrelate			= sfle.d( pE, fileDirSrc, "correlate.py" )
c_fileProgCorrs2Heatmap		= sfle.d( pE, fileDirSrc, "corrs2heatmap.py" )

#===============================================================================
# Metaphlan annotated PCL
#===============================================================================

sfle.pipe( pE, c_fileInputMetaphlanPCL, c_fileProgMetaphlan2PCL, c_fileMetaphlanPrePCL )
sfle.pipe( pE, c_fileMetaphlanPrePCL, c_fileProgMergeMetadata, c_fileMetaphlanPCL,
	[[False, c_fileInputMetadataTXT], "-t 0", "-n"] )
sfle.pipe( pE, c_fileMetaphlanPCL, c_fileProgTranspose, c_fileMetaphlanTSV )
sfle.spipe( pE, c_fileMetaphlanTSV, "grep -P '(sample)|(Tongue_dorsum)'", c_fileTDPreTSV )
sfle.pipe( pE, c_fileTDPreTSV, c_fileProgGrepRows, c_fileTDTSV, [[False, c_fileInputExcludeDAT], "-f"] )
sfle.pipe( pE, c_fileTDTSV, c_fileProgTranspose, c_fileTDPCL )

#===============================================================================
# KO abundances annotated PCL
#===============================================================================

sfle.spipe( pE, c_fileInputKOsPCL, "cut -f2- | sed -r 's/_vs_\\S+//g' | grep -P '^((NAME)|(K\\d+))'", c_fileKOsPrePCL )
sfle.pipe( pE, c_fileKOsPrePCL, c_fileProgMergeMetadata, c_fileKOsPCL,
	[[False, c_fileInputMetadataTXT], "-t 0", "-n"] )
sfle.pipe( pE, c_fileKOsPCL, c_fileProgTranspose, c_fileKOsTSV )
sfle.spipe( pE, c_fileKOsTSV, "grep -P '(sample)|(Tongue_dorsum)'", c_fileTDKOsPreTSV )
sfle.pipe( pE, c_fileTDKOsPreTSV, c_fileProgGrepRows, c_fileTDKOsTSV, [[False, c_fileInputExcludeDAT], "-f"] )
sfle.pipe( pE, c_fileTDKOsTSV, c_fileProgTranspose, c_fileTDKOsPCL )

#===============================================================================
# Abundance barcharts
#===============================================================================

def funcImg( astrArgs = [] ):
	def funcRet( target, source, env, astrArgs = astrArgs ):
		astrTs, astrSs = ([f.get_abspath( ) for f in a] for a in (target, source))
		strProg, strIn = astrSs[:2]
		return sfle.ex( [sfle.cat( strIn ), "|", strProg, astrTs[0]] + astrSs[2:] + astrArgs,
			astrTs[1] if ( len( astrTs ) > 1 ) else None )
	return funcRet

Command( [c_fileMetaphlanPDF, c_fileBugsTXT], [c_fileProgPCL2Img, c_fileTDPCL], funcImg( [c_strMetadatum] ) )
Command( c_fileOTUsPDF, [c_fileProgPCL2Img, c_fileInputOTUsPCL], funcImg( [c_strMetadatum] ) )

#===============================================================================
# Carriage curves and modules heatmap
#===============================================================================

Command( c_fileCarriagePDF, [c_fileProgPCL2CDF, c_fileTDPCL], funcImg( ) )

sfle.pipe( pE, c_fileInputIMGPCL, c_fileProgPathcov, c_fileModulesPCL,
	[[False, s] for s in (c_fileInputModuleP, c_fileInputMapKEGGTXT)] )

Command( c_fileModulesPDF, [c_fileProgModules2Heatmap, c_fileModulesPCL, c_fileBugsTXT], funcImg( ) )

#===============================================================================
# Correlate KOs and pathogens
#===============================================================================

sfle.pipe( pE, c_fileTDKOsPCL, c_fileProgCorrelate, c_fileCorrelationsTXT,
	[[False, c_fileTDPCL], c_strMetadatum] )
Command( c_fileCorrelationsPDF, [c_fileProgCorrs2Heatmap, c_fileCorrelationsTXT, c_fileInputPathogensTXT],
	funcImg( ) )

Default( (c_fileMetaphlanPDF, c_fileOTUsPDF, c_fileCarriagePDF, c_fileModulesPDF, c_fileCorrelationsPDF) )
