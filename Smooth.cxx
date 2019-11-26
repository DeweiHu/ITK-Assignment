#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCurvatureAnisotropicDiffusionImageFilter.h"
#include <iostream>
#include <glob.h>
#include <vector>

// Directory
const char* PATH_CNV = "/home/dewei/Desktop/cnv_selected/*";
const char* PATH_DME = "/home/dewei/Desktop/dme_selected/*";
const char* PATH_DRUSEN = "/home/dewei/Desktop/drusen_selected/*";
const char* PATH_NORMAL = "/home/dewei/Desktop/normal_selected/*";

const char* PATH_save = "/home/dewei/Desktop/normal_anisotropic/";

const unsigned int Dimension = 2;
typedef itk::Image< float, Dimension > ImageType;

int main(int argc, char * argv[])
{
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typedef itk::ImageFileWriter<ImageType> WriterType;
    typedef itk::CurvatureAnisotropicDiffusionImageFilter < ImageType, ImageType > SmoothFilterType;

    const unsigned int numberOfIterations = 50;
    // Read all filenames under directory
    glob_t glob_result;
    glob(PATH_NORMAL,GLOB_TILDE,NULL, &glob_result);

    char* Name;
    int iter = 500;
    if(glob_result.gl_pathc<500){
        iter = glob_result.gl_pathc;
    }
    for(unsigned int i=0; i<iter;++i){
        Name = glob_result.gl_pathv[i];

        // Read image by name
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName( Name );
        reader->Update();

        // Anisotropic Smoothing filter
        SmoothFilterType::Pointer  smoothfilter = SmoothFilterType::New();
        smoothfilter->SetInput( reader->GetOutput() );
        smoothfilter->SetNumberOfIterations( numberOfIterations );
        smoothfilter->Update();

        // write output
        std::string str = std::to_string(i);
        std::cout << "Number: " << i << " is under processing" << std::endl;

        WriterType::Pointer writer = WriterType::New();
        writer->SetInput( smoothfilter->GetOutput() );
        writer->SetFileName( PATH_save+str+".nii" );
        writer->Update();
    }

    return 0;
}