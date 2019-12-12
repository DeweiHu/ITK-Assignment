#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBayesianClassifierInitializationImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkThresholdImageFilter.h"

#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"


const char* Input_path = "/home/dewei/Desktop/final/4.nii.gz";
const char* Output_path = "/home/dewei/Desktop/final/label_4.nii";
const char* WM_path = "/home/dewei/Desktop/final/WM_4.nii";

const unsigned int Dimension = 3;
const unsigned int ClassNum = 5;
const unsigned int iter = 1;

typedef float PixelType;
typedef itk::Image< PixelType,Dimension > ImageType;
typedef itk::VectorImage< PixelType,Dimension > InputImageType;
typedef itk::Image< unsigned char,Dimension > OutputImageType;

int main() {
    // Reader
    typedef itk::ImageFileReader< ImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( Input_path );
    reader->Update();

    // Pre-thresholding to clear the part around the brain
    unsigned char thresh = 170;
    typedef itk::ThresholdImageFilter< ImageType > PreThresholdFilterType;
    PreThresholdFilterType::Pointer preFilter = PreThresholdFilterType::New();
    preFilter->SetInput( reader->GetOutput() );
    preFilter->ThresholdBelow( thresh );
    preFilter->SetOutsideValue( 0 );


    // Classifier Initializer
    typedef itk::BayesianClassifierInitializationImageFilter< ImageType > BayesianInitializerType;
    BayesianInitializerType::Pointer bayesianInitializer = BayesianInitializerType::New();
    bayesianInitializer->SetInput( preFilter->GetOutput() );
    bayesianInitializer->SetNumberOfClasses( ClassNum );
    bayesianInitializer->Update();

    // Classifier and Gaussian filter
    typedef unsigned char LabelType;
    typedef float PriorType;
    typedef float PosteriorType;

    typedef itk::BayesianClassifierImageFilter< InputImageType, LabelType, PosteriorType >\
    ClassifierFilterType;
    ClassifierFilterType::Pointer Classifierfilter = ClassifierFilterType::New();
    Classifierfilter->SetInput( bayesianInitializer->GetOutput() );
    Classifierfilter->SetNumberOfSmoothingIterations( iter );

    typedef ClassifierFilterType::ExtractedComponentImageType ECImageType;
    typedef itk::SmoothingRecursiveGaussianImageFilter< ECImageType,ECImageType > SmoothingFilterType;
    SmoothingFilterType::Pointer smoother = SmoothingFilterType::New();
    smoother->SetSigma( 1 );
    Classifierfilter->SetSmoothingFilter( smoother );

    typedef ClassifierFilterType::OutputImageType ClassifierOutputImageType;
    typedef itk::RescaleIntensityImageFilter< ClassifierOutputImageType,OutputImageType > RescalerType;
    RescalerType::Pointer rescaler = RescalerType::New();
    rescaler->SetInput( Classifierfilter->GetOutput() );
    rescaler->SetOutputMinimum( 0 );
    rescaler->SetOutputMaximum( 255 );

    // Binary thresholding
    typedef itk::BinaryThresholdImageFilter< OutputImageType,OutputImageType > ThresholdFilterType;
    ThresholdFilterType::Pointer Thresholdfilter = ThresholdFilterType::New();
    Thresholdfilter->SetInput( rescaler->GetOutput() );

    const unsigned char outsideValue = 0;
    const unsigned char insideValue = 255;
    Thresholdfilter->SetOutsideValue( outsideValue );
    Thresholdfilter->SetInsideValue( insideValue );

    const unsigned char lowerThreshold = 190;
    const unsigned char upperThreshold = 195;
    Thresholdfilter->SetLowerThreshold( lowerThreshold );
    Thresholdfilter->SetUpperThreshold( upperThreshold );

    Thresholdfilter->Update();

    // Morphological Opening Operater
    unsigned int radius = 1;
    typedef itk::BinaryBallStructuringElement< OutputImageType::PixelType, Dimension > StructuringElementType;
    StructuringElementType structuringElement;
    structuringElement.SetRadius( radius );
    structuringElement.CreateStructuringElement();

    typedef itk::BinaryMorphologicalOpeningImageFilter< OutputImageType,OutputImageType,StructuringElementType >\
    OpeningFilterType;
    OpeningFilterType::Pointer openingFilter = OpeningFilterType::New();
    openingFilter->SetInput( Thresholdfilter->GetOutput() );
    openingFilter->SetKernel( structuringElement );
    openingFilter->Update();

    // Get largest connected component
    typedef unsigned int Label;
    typedef itk::Image< Label,Dimension > LabelImageType;

    typedef itk::ConnectedComponentImageFilter< OutputImageType, LabelImageType > ConnectedComponentFilterType;
    ConnectedComponentFilterType::Pointer connected = ConnectedComponentFilterType::New();
    connected->SetInput( openingFilter->GetOutput() );
    connected->Update();

    typedef itk::LabelShapeKeepNObjectsImageFilter< LabelImageType > ShapeKeepFilterType;
    ShapeKeepFilterType::Pointer keeper = ShapeKeepFilterType::New();
    keeper->SetInput( connected->GetOutput() );
    keeper->SetBackgroundValue( 0 );
    keeper->SetNumberOfObjects( 1 );
    keeper->SetAttribute( ShapeKeepFilterType::LabelObjectType::NUMBER_OF_PIXELS );

    typedef itk::RescaleIntensityImageFilter< LabelImageType,ImageType > RescaleFilterType;
    RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetOutputMinimum( 0 );
    rescaleFilter->SetOutputMaximum( 255 );
    rescaleFilter->SetInput( keeper->GetOutput() );

// Writer
    typedef  itk::ImageFileWriter< OutputImageType > WriterType;
    WriterType::Pointer writer1 = WriterType::New();
    writer1->SetFileName( Output_path );
    writer1->SetInput( rescaler->GetOutput() );
    writer1->Update();

    typedef itk::ImageFileWriter< ImageType > WriterType2;
    WriterType2::Pointer writer2 = WriterType2::New();
    writer2->SetFileName( WM_path );
    writer2->SetInput( rescaleFilter->GetOutput() );
    writer2->Update();

    return 0;
}