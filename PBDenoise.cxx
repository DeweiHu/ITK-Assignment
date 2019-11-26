#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGaussianRandomSpatialNeighborSubsampler.h"
#include "itkPatchBasedDenoisingImageFilter.h"

#include <string>
#include <iostream>

const char* PATH_in = "/home/dewei/Desktop/cnv_selected/1.jpeg";
const char* PATH_out = "/home/dewei/Desktop/cnv_denoised/1.nii";

int main( int argc, char *argv[] )
{
   typedef float PixelType;
   const unsigned int Dimension = 2;

   typedef itk::Image < PixelType, Dimension > ImageType;
   typedef itk::PatchBasedDenoisingImageFilter< ImageType, ImageType > FilterType;
   typedef typename FilterType::PatchWeightsType PatchType;
   typedef itk::Statistics::GaussianRandomSpatialNeighborSubsampler\
   < typename FilterType::PatchSampleType, typename ImageType::RegionType > SamplerType;

   // read the noisy image to be denoised
   typedef itk::ImageFileReader < ImageType > ReaderType;
   ReaderType::Pointer reader = ReaderType::New();
   reader->SetFileName( PATH_in );
   reader->Update();

   // Filter settings and initialization

   FilterType::Pointer filter = FilterType::New();
   filter->SetInput( reader->GetOutput() );

   // patch radius is same for all dimensions of the image
   unsigned int patchRadius = 4;
   filter->SetPatchRadius( patchRadius );

   // instead of directly setting the weights, could also specify type
   filter->UseSmoothDiscPatchWeightsOn();
   filter->UseFastTensorComputationsOn();

   std::string noiseModel = "gaussian";
   // 3 types of noise model can be applied:
   if( noiseModel=="gaussian" ){
       filter->SetNoiseModel( FilterType::GAUSSIAN );
   }
   else if( noiseModel=="rician" ){
       filter->SetNoiseModel( FilterType::RICIAN );
   }
   else if( noiseModel=="poisson" ){
       filter->SetNoiseModel( FilterType::POISSON );
   }

   // step size or weight for smoothing term, large step sizes may cause instabilities.
   // step size or weight for fidelity term
   // use a positive weight to prevent oversmoothing (penalizes deviations from noisy data based on a noise model)
   filter->SetSmoothingWeight( 1 );
   float fidelityWeight = 1;
   filter->SetSmoothingWeight( fidelityWeight );

   // number of iterations over the image of denoising
   unsigned int numIterations = 3;
   filter->SetNumberOfIterations( numIterations );
/*
   // sampling the image to find similar patches
   typename SamplerType::Pointer sampler = SamplerType::New();
   // variance (in physical units) for semi-local Gaussian sampling
   sampler->SetVariance( 400 );
   // rectangular window restricting the Gaussian sampling
   sampler->SetRadius( 50 ); // 2.5 * standard deviation
   // number of random sample "patches" to use for computations
   sampler->SetNumberOfResultsRequested( 500 );
   // Sampler can be complete neighborhood sampler, random neighborhood sampler, Gaussian sampler, etc.
   filter->SetSampler( sampler );

   // automatic estimation of the kernel bandwidth
   filter->KernelBandwidthEstimationOn();
   // update bandwidth every 'n' iterations
   filter->SetKernelBandwidthUpdateFrequency( 3 );
   // use 33% of the pixels for the sigma update calculation
   filter->SetKernelBandwidthFractionPixelsForEstimation( 0.20 );

   // multiplication factor modifying the automatically-estimated kernel sigma
   float sigmaMultiplicationFactor = 1.0;
   filter->SetKernelBandwidthMultiplicationFactor(sigmaMultiplicationFactor);
*/
   // Denoise the image
   std::cout << "Filter prior to update:\n";
   filter->Print( std::cout );
   try
     {
       filter->Update();
     }
   catch( itk::ExceptionObject & excp )
     {
       std::cout << "Error: In " __FILE__ ", line " << __LINE__ << "\n"
            << "Caught exception <" << excp
            << "> while running patch-based denoising image filter."
            << "\n\n";
     return EXIT_FAILURE;
     }

   filter->GetOutput()->Print( std::cout, 3 );

   // write the denoised image to file
   typedef typename FilterType::OutputImageType OutputImageType;

   typedef itk::ImageFileWriter< OutputImageType > WriterType;
   WriterType::Pointer writer = WriterType::New();
   writer->SetFileName( PATH_out );
   writer->SetInput( filter->GetOutput() );
   writer->Update();

   return 0;
}


