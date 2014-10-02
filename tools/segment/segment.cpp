/*
 Copyright (C) 2006 Pedro Felzenszwalb

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#include <stdio.h>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"
#include <string>
#include <iostream>

using namespace std;

void saveComps(const char *fn, map<int, vecInt*> &comps) {
//	ofstream fp(fn);
	FILE* fp = fopen(fn, "w");
//	printf("%d\n",comps.size());
//	fp<<comps.size()<<endl;
	fprintf(fp, "%d\n", comps.size());
	for (map<int, vecInt*>::iterator it = comps.begin(); it != comps.end();
			++it) {
//		fp<<it->second->size();
		fprintf(fp, "%d", it->second->size());
//		printf("%d",it->second->size());
		for (int j = 0; j < it->second->size(); ++j) {
//			fp<<" "<<(*it->second)[j];
			fprintf(fp, " %d", (*it->second)[j]);
//			printf(" %d",(*it->second)[j]);
		}
//		printf("\n");
		fprintf(fp, "\n");
//		fp<<endl;
	}
//	fp.close();
	fclose(fp);
}

int main(int argc, char **argv) {
	if (argc != 7) {
		fprintf(stderr, "need 6 parameters. %d are given\n", argc - 1);
		fprintf(stderr, "usage: %s sigma k min max input(ppm) output(ppm)\n",
				argv[0]);
		return 1;
	}
	float sigma = atof(argv[1]);
	float k = atof(argv[2]);
	int min_size = atoi(argv[3]);
	int max_size = atoi(argv[4]);

	printf("loading input image %s\n", argv[5]);
	image<rgb> *input = loadPPM(argv[5]);
	string compResFileNm = string(argv[5]);
	compResFileNm.erase(compResFileNm.end() - 4, compResFileNm.end());
	compResFileNm = compResFileNm + string(".seg");

	printf("processing\n");
	int num_ccs;
	map<int, vecInt*> comps;
	image<rgb> *seg = segment_image(input, sigma, k, min_size, max_size,
			&num_ccs, comps);
	savePPM(seg, argv[6]);

	printf("save segmentation results to file %s\n", compResFileNm.c_str());
	saveComps(compResFileNm.c_str(), comps);

//  printf("clean memory\n");
//  for(map<int,vecInt*>::iterator it=comps.begin();it!=comps.end();++it){
//	  delete it->second;
//  }
//  comps.clear();
//  printf("finish memory clean\n");
//
//  printf("got %d components\n", num_ccs);
//  printf("done! uff...thats hard work.\n");

	return 0;
}

