#include <iostream>
#include <glob.h>
#include <vector>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"

// Directory of images
const char* PATH = "/home/dewei/Desktop/itkAssignment/AffineResults/*";

// IMAGE TYPE: Define the PixelType and image dimension for the image, using keyword [ImageType]
typedef float PixelType;
const unsigned int Dimension = 3;
typedef itk::Image < PixelType, Dimension > ImageType; //define the image type

// Define the function
ImageType::Pointer ImageSum( ImageType::Pointer buffer, char* FileName ){
    // Setup image reader
    typedef itk::ImageFileReader< ImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( FileName );
    reader->Update();

    // Pixel-wise add 2 images
    typedef itk::AddImageFilter< ImageType,ImageType,ImageType > AddImageFilterType;
    AddImageFilterType::Pointer addImageFilter = AddImageFilterType::New();
    addImageFilter->SetInput1(buffer);
    addImageFilter->SetInput2(reader->GetOutput());
    addImageFilter->Update();

    return addImageFilter->GetOutput();
}

int main (int argc, char* argv[]) {
  // Read all the filenames
  glob_t glob_result;
  glob(PATH,GLOB_TILDE,NULL, &glob_result);

  char* NameList[glob_result.gl_pathc];
  for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
      NameList[i] = glob_result.gl_pathv[i];
  }

  // Initial the buffer by image[0]
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer reader0 = ReaderType::New();
  reader0->SetFileName( NameList[0] );
  reader0->Update();
  ImageType::Pointer buffer = reader0->GetOutput();

  // loop to add other images to the buffer
  for(unsigned int i=1; i<glob_result.gl_pathc; ++i){
      buffer = ImageSum(buffer,NameList[i]);
  }

  // Get the average by divide the sum by the number of image
  const float factor = 0.05;
  typedef itk::MultiplyImageFilter < ImageType,ImageType,ImageType > MultiplyFilterType;
  MultiplyFilterType::Pointer multiplyFilter = MultiplyFilterType::New();
  multiplyFilter->SetInput1(buffer);
  multiplyFilter->SetConstant2(factor);
  multiplyFilter->Update();

  // Write out the result to argv[1]
  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType:: Pointer writer = WriterType::New();
  writer->SetFileName( argv[1] );
  writer->SetInput( multiplyFilter->GetOutput() );
  writer->Update();

  return 0;
}
