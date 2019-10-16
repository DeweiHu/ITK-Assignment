#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include <iostream>
#include <glob.h>
#include <vector>
#include <string>

// Directory of images
const char* PATH = "/home/dewei/Desktop/itkAssignment/data/*";

// Setup types
const unsigned int Dimension = 3 ;
typedef float PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;

// Affine Registration Function
ImageType::Pointer AffineRegistration( ImageType::Pointer fixedImage, char* Filename ){
    ////////////////////////////  SETUP  ///////////////////////////////////////
    // Create and setup a reader for moving image
    typedef itk::ImageFileReader< ImageType >  readerType;
    readerType::Pointer reader1 = readerType::New();
    reader1->SetFileName( Filename );
    reader1->Update();
    ImageType::Pointer movingImage = reader1->GetOutput() ;

    // Define types for 1.Transform  2.Metric  3.Interpolation  4.Gradient Descent  5.Registration Wrapper
    typedef itk::AffineTransform < double, 3 > TransformType ;
    typedef itk::ImageRegistrationMethod < ImageType, ImageType > RegistrationWrapperType ;  // fixed, moving
    typedef itk::MeanSquaresImageToImageMetric < ImageType, ImageType > MetricType ; // fixed, moving
    typedef itk::LinearInterpolateImageFunction < ImageType, double > InterpolatorType ; // moving, coordinate representation
    typedef itk::RegularStepGradientDescentOptimizer OptimizerType ;

    // Define corresponding pointers
    RegistrationWrapperType::Pointer registrationWrapper = RegistrationWrapperType::New() ;
    TransformType::Pointer transform = TransformType::New() ;
    MetricType::Pointer metric = MetricType::New() ;
    InterpolatorType::Pointer interpolator = InterpolatorType::New() ;
    OptimizerType::Pointer optimizer = OptimizerType::New() ;
    optimizer->SetMaximumStepLength( 0.125 );

    // Connect the pipeline and Initialization
    transform->SetIdentity() ;
    registrationWrapper->SetMovingImage ( movingImage ) ;
    registrationWrapper->SetFixedImage ( fixedImage ) ;
    registrationWrapper->SetTransform ( transform ) ;
    registrationWrapper->SetMetric ( metric ) ;
    registrationWrapper->SetInterpolator ( interpolator ) ;
    registrationWrapper->SetOptimizer ( optimizer ) ;
    registrationWrapper->SetInitialTransformParameters ( transform->GetParameters() ) ;
    registrationWrapper->SetFixedImageRegion ( fixedImage->GetLargestPossibleRegion() ) ;
    ///////////////////////////  DO REGISTRATION  ///////////////////////////////
    // Run the registration
    try
    {
        registrationWrapper->Update() ;
    }
    catch ( itk::ExceptionObject & excp )
    {
        std::cerr << "Error in registration" << std::endl;
        std::cerr << excp << std::endl;
    }
    // Update the transform
    transform->SetParameters ( registrationWrapper->GetLastTransformParameters() ) ;
    // Apply the transform
    typedef itk::ResampleImageFilter < ImageType, ImageType > ResampleFilterType ;
    ResampleFilterType::Pointer filter = ResampleFilterType::New() ;
    filter->SetInput ( movingImage ) ;
    //filter->SetTransform ( dynamic_cast < const TransformType * > ( registrationWrapper->GetOutput() ) ) ;
    filter->SetTransform ( transform ) ;
    filter->SetSize ( movingImage->GetLargestPossibleRegion().GetSize() ) ;
    filter->SetReferenceImage ( fixedImage ) ;
    filter->UseReferenceImageOn() ;
    filter->Update() ;
    return filter->GetOutput();
}

int main(){
    // Save all the filenames
    glob_t glob_result;
    glob(PATH,GLOB_TILDE,NULL, &glob_result);
    char* NameList[glob_result.gl_pathc];
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        NameList[i] = glob_result.gl_pathv[i];
    }
    // Same for the fixed image
    typedef itk::ImageFileReader < ImageType > ReaderType;
    ReaderType::Pointer reader2 = ReaderType::New() ;
    reader2->SetFileName ( NameList[0] ) ;
    reader2->Update() ;
    ImageType::Pointer fixedImage = reader2->GetOutput() ;

    // Define the writer
    typedef  itk::ImageFileWriter < ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();

    const char* head = "/home/dewei/Desktop/itkAssignment/AffineResults/Affine_";

    for(unsigned int i=1; i<glob_result.gl_pathc; ++i){

        std::cout << "Item:" << i << std::endl;
        std::string str = std::to_string(i);

        ImageType::Pointer output = AffineRegistration(fixedImage,NameList[i]);
        writer->SetInput( output );
        writer->SetFileName ( head+str+".nii" );
        writer->Update();

    }
    return 0 ;
}
