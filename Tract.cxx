#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNrrdImageIO.h"

#include "itkVector.h"
#include "itkVectorImage.h"
#include "itkArray2D.h"
#include "itkImageRegionIterator.h"

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

// Directory of image
const char* PATH = "/home/dewei/Desktop/Tractography/DTIBrain.nrrd";
const char* OutputName = "/home/dewei/Desktop/Eigenvectors.nrrd";

typedef float ComponentType;
const unsigned int ImageDimension = 3;
typedef itk::Vector< ComponentType,9 > VoxelType;
typedef itk::Vector< ComponentType,3 > VectorType;
typedef itk::Image< VoxelType ,ImageDimension > ImageType;
typedef itk::Image< VectorType,ImageDimension > OutputImageType;

typedef itk::ImageFileReader< ImageType > readerType;
typedef itk::ImageFileWriter< OutputImageType > writerType;

VectorType GetMaxEigenVector(const ImageType::IndexType voxelIndex, ImageType::Pointer InputImage){
    ImageType::PixelType tensor = InputImage->GetPixel( voxelIndex );
//    std::cout << "tensor: " << tensor << std::endl;
    typedef itk::Matrix< ComponentType,3,3 > InputMatrixType;
    typedef itk::FixedArray< ComponentType,3 > EigenValueArrayType;
    typedef itk::Matrix< ComponentType,3,3 > EigenVectorMatrixType;
    typedef itk::SymmetricEigenAnalysisFixedDimension< 3, InputMatrixType,EigenValueArrayType,\
    EigenVectorMatrixType > SymmetricEigenAnalysisType;

    InputMatrixType  cov;
    EigenValueArrayType  eigenvalues;
    EigenVectorMatrixType  eigenvectors;
    SymmetricEigenAnalysisType symmetricEigenSystem;

    // covariance matrix: 1*9 array to 3*3 matrix
    for(unsigned int row=0; row<3; row++){
        for(unsigned int col=0; col<3; col++){
            cov[row][col] = tensor[row*3+col];
        }
    }
    symmetricEigenSystem.SetOrderEigenMagnitudes( true );
    symmetricEigenSystem.ComputeEigenValuesAndVectors( cov,eigenvalues,eigenvectors );
/*    std::cout << "EigenValues: " << eigenvalues << std::endl;
    std::cout << "EigenVectors (each row): " << std::endl;
    std::cout << eigenvectors << std::endl;
*/
    VectorType MaxEigenVector;
    MaxEigenVector[0] = eigenvectors[2][0];
    MaxEigenVector[1] = eigenvectors[2][1];
    MaxEigenVector[2] = eigenvectors[2][2];

//    std::cout << "max eigenvector: " << MaxEigenVector << std::endl;

    return MaxEigenVector;
}


int main(){
    readerType::Pointer reader = readerType::New();
    reader->SetFileName( PATH );
    reader->Update();

    itk::NrrdImageIO::Pointer io = itk::NrrdImageIO::New();
    io->SetFileType( itk::ImageIOBase::ASCII );

    ImageType::Pointer InputImage = ImageType::New();
    InputImage = reader->GetOutput();

    typedef itk::ImageRegionIterator < OutputImageType > IteratorType;
    ImageType::RegionType region_in = InputImage->GetLargestPossibleRegion();
    ImageType::SizeType Size_in = region_in.GetSize();
    ImageType::IndexType Corner_in = region_in.GetIndex();

    std::cout << "region: " << region_in << std::endl;
    std::cout << "Size: " << Size_in << std::endl;
    std::cout << "Corner: " << Corner_in << std::endl;

    // Initialize the output image as a buffer
    OutputImageType::Pointer OutputImage = OutputImageType::New();
    const OutputImageType::IndexType start = Corner_in;
    const OutputImageType::SizeType Size_out = Size_in;

    OutputImageType::RegionType region_out;
    region_out.SetSize( Size_out );
    region_out.SetIndex( start );

    OutputImage->SetRegions( region_out );
//    OutputImage->SetVectorLength(3);
    OutputImage->Allocate();

    // Use iterator to go though all voxels to compute the max eigenvector
    IteratorType iterator( OutputImage,region_in );
    iterator.GoToBegin();
    while( ! iterator.IsAtEnd() ){
        ImageType::IndexType voxelIndex = iterator.GetIndex();
        VectorType MaxEigenVector;
        MaxEigenVector = GetMaxEigenVector( voxelIndex,InputImage );
        if(MaxEigenVector[2]==1){
            MaxEigenVector[2] = 0;
        }
        iterator.Set( MaxEigenVector );
        std::cout << iterator.GetIndex() << ":  " << MaxEigenVector << std::endl;
        ++iterator;
    }

    writerType::Pointer writer = writerType::New();
    writer->UseInputMetaDataDictionaryOn();
    writer->SetInput( OutputImage );
    writer->SetImageIO( io );
    writer->SetFileName( OutputName );
    writer->Update();

    return 0 ;
}
