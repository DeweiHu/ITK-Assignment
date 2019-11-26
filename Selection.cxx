#include <iostream>
#include <glob.h>
#include <vector>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

// Directory
const char* PATH_CNV = "/home/dewei/Desktop/data/OCT2017 /train/CNV/*";
const char* PATH_DME = "/home/dewei/Desktop/data/OCT2017 /train/DME/*";
const char* PATH_DRUSEN = "/home/dewei/Desktop/data/OCT2017 /train/DRUSEN/*";
const char* PATH_NORMAL = "/home/dewei/Desktop/data/OCT2017 /train/NORMAL/*";

const char* PATH_save = "/home/dewei/Desktop/normal_selected/";

// Image Type define
typedef unsigned char PixelType;
const unsigned int Dimension = 2;
typedef itk::Image < PixelType, Dimension > ImageType;

int main(int argc, char* argv[]){

    // Read all filenames under directory
    glob_t glob_result;
    glob(PATH_NORMAL, GLOB_TILDE, NULL, &glob_result);

    char* Name;
    int cnt = 0;
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        Name = glob_result.gl_pathv[i];
        // Image Reader
        typedef itk::ImageFileReader < ImageType > ReaderType;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName( Name );
        reader->Update();

        ImageType::Pointer image = reader->GetOutput();
        ImageType::RegionType region = image->GetLargestPossibleRegion();
        ImageType::SizeType size = region.GetSize();

        if(size[0]==1536 && size[1]==496){
            cnt ++;
            std::string str = std::to_string(cnt);
            std::cout << cnt << ' ' << Name << " " << size << std::endl;

            typedef itk::ImageFileWriter < ImageType > WriterType;
            WriterType::Pointer writer = WriterType::New();
            writer->SetFileName(PATH_save+str+".jpeg");
            writer->SetInput( image );
            writer->Update();

        }
    }

    return 0;
}