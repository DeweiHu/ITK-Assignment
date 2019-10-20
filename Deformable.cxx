#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSymmetricForcesDemonsRegistrationFilter.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkWarpImageFilter.h"

const char* FixName = "/home/dewei/Desktop/itkAssignment/AffineAverage.nii.gz";
const unsigned int Dimension = 3;
typedef float PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;


class CommandIterationUpdate : public itk::Command
{
public:
    typedef CommandIterationUpdate Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer < Self > Pointer;
    itkNewMacro( CommandIterationUpdate );
protected:
    CommandIterationUpdate() {};
    typedef itk::Image < PixelType, Dimension > ImageType;
    typedef itk::Vector < PixelType, Dimension > VectorPixelType;
    typedef itk::Image < VectorPixelType, Dimension > DisplacementFieldType;
    typedef itk::SymmetricForcesDemonsRegistrationFilter < ImageType, ImageType, DisplacementFieldType > RegistrationFilterType;
public:
    void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
        Execute( (const itk::Object *)caller, event);
    }
    void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
        const RegistrationFilterType * filter = static_cast< const RegistrationFilterType * >( object );
        if( !(itk::IterationEvent().CheckEvent( &event )) )
        {
            return;
        }
        std::cout << filter->GetMetric() << std::endl;
    }
};

int main( int argc, char *argv[] )
{

    typedef itk::ImageFileReader < ImageType > ReaderType;
    ReaderType::Pointer fixedImageReader = ReaderType::New();
    ReaderType::Pointer movingImageReader = ReaderType::New();
    fixedImageReader->SetFileName( FixName );
    movingImageReader->SetFileName( argv[1] );

    typedef  itk::CastImageFilter < ImageType, ImageType > CasterType;
    CasterType::Pointer fixedImageCaster = CasterType::New();
    CasterType::Pointer movingImageCaster = CasterType::New();
    fixedImageCaster->SetInput( fixedImageReader->GetOutput() );
    movingImageCaster->SetInput( movingImageReader->GetOutput() );

    typedef itk::HistogramMatchingImageFilter < ImageType, ImageType > MatchingFilterType;
    MatchingFilterType::Pointer matcher = MatchingFilterType::New();

    // For this example, we set the moving image as the source or input image and
    // the fixed image as the reference image.
    matcher->SetInput( movingImageCaster->GetOutput() );
    matcher->SetReferenceImage( fixedImageCaster->GetOutput() );

    // We then select the number of bins to represent the histograms and the
    // number of points or quantile values where the histogram is to be
    // matched.
    matcher->SetNumberOfHistogramLevels( 1024 );
    matcher->SetNumberOfMatchPoints( 7 );

    // Simple background extraction is done by thresholding at the mean
    // intensity.
    matcher->ThresholdAtMeanIntensityOn();

    typedef itk::Vector < float,Dimension > VectorPixelType;
    typedef itk::Image < VectorPixelType, Dimension > DisplacementFieldType;
    typedef itk::SymmetricForcesDemonsRegistrationFilter < ImageType, ImageType, DisplacementFieldType > RegistrationFilterType;

    RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    filter->AddObserver( itk::IterationEvent(), observer );

    filter->SetFixedImage( fixedImageCaster->GetOutput() );
    filter->SetMovingImage( matcher->GetOutput() );

    // The demons registration filter has two parameters: the number of
    // iterations to be performed and the standard deviation of the Gaussian
    // smoothing kernel to be applied to the deformation field after each
    filter->SetNumberOfIterations( 150 );
    filter->SetStandardDeviations( 1.0 );
    filter->Update();

    typedef itk::WarpImageFilter < ImageType, ImageType, DisplacementFieldType > WarperType;
    typedef itk::LinearInterpolateImageFunction < ImageType, double > InterpolatorType;
    WarperType::Pointer warper = WarperType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    ImageType::Pointer fixedImage = fixedImageReader->GetOutput();
    warper->SetInput( movingImageReader->GetOutput() );
    warper->SetInterpolator( interpolator );
    warper->SetOutputSpacing( fixedImage->GetSpacing() );
    warper->SetOutputOrigin( fixedImage->GetOrigin() );
    warper->SetOutputDirection( fixedImage->GetDirection() );

    // Unlike the ResampleImageFilter, the WarpImageFilter
    // warps or transform the input image with respect to the deformation field
    // represented by an image of vectors.  The resulting warped or resampled
    // image is written to file as per previous examples.
    CasterType::Pointer caster = CasterType::New();
    warper->SetDisplacementField( filter->GetOutput() );
    caster->SetInput( warper->GetOutput() );

    typedef itk::ImageFileWriter < ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput( caster->GetOutput() );
    writer->SetFileName ( argv[2] );
    writer->Update();

    return EXIT_SUCCESS;
}