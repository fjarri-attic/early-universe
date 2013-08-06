#!/bin/bash

opt="vbitrate=2160000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq"

mencoder -ovc lavc -lavcopts vcodec=mpeg4:vpass=1:$opt -mf type=png:fps=10 -nosound -o /dev/null $1
mencoder -ovc lavc -lavcopts vcodec=mpeg4:vpass=2:$opt -mf type=png:fps=10 -nosound -o $2 $1
#mencoder -ovc lavc -lavcopts vcodec=mpeg4:$opt -mf type=png:fps=10 -nosound -o $2 $1
