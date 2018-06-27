#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "itkMaskNegatedImageFilter.h"
#include "itkConnectedThresholdImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMergeLabelMapFilter.h"
#include "itkLabelMapToBinaryImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageDuplicator.h"
#include "itkGrayscaleFillholeImageFilter.h"

#include <vector>
#include "cipChestConventions.h"
#include "cipHelper.h"

#include "SegmentLungAirwaysCLP.h"

#define DIM 3

/** FUNCTION FOR TRACHEA SEGMENTATION */
cip::LabelMapType::Pointer TracheaSegmentation( cip::CTType::Pointer VOI,
                                                cip::CTType::IndexType indexFiducialSlice,
                                                std::vector< std::vector<float> > fiducial
)
{
    cip::LabelMapType::Pointer trachea 		= cip::LabelMapType::New();
    cip::LabelMapType::Pointer tracheaPrev 	= cip::LabelMapType::New();

    trachea->SetRegions( VOI->GetRequestedRegion() );
    trachea->SetBufferedRegion( VOI->GetBufferedRegion() );
    trachea->SetLargestPossibleRegion( VOI->GetLargestPossibleRegion() );
    trachea->CopyInformation( VOI );
    trachea->Allocate();

    /** TRACHEA SEGMENTATION PIPELINE */
    typedef itk::ConnectedThresholdImageFilter< cip::CTType, cip::CTType > ConnectedFilterType;
    ConnectedFilterType::Pointer thresholdConnected = ConnectedFilterType::New();

    thresholdConnected->SetInput( VOI );
    thresholdConnected->SetReplaceValue( cip::AIRWAY );

    // Starting upper threshold value
    signed short UpperThreshold = -900;

    thresholdConnected->SetUpper( UpperThreshold );

    cip::CTType::PointType lpsPoint;
    cip::CTType::IndexType index;

    // Seeds come in ras, convert to lps
    for( ::size_t i = 0; i < fiducial.size(); ++i )
    {
        lpsPoint[0] = fiducial[0][0] * ( -VOI->GetDirection()[0][0] );
        lpsPoint[1] = fiducial[0][1] * ( -VOI->GetDirection()[1][1] );
        lpsPoint[2] = fiducial[0][2] *   VOI->GetDirection()[2][2];

        VOI->TransformPhysicalPointToIndex(lpsPoint, index);
        thresholdConnected->AddSeed( index );
    }

    typedef itk::CastImageFilter<cip::CTType, cip::LabelMapType> CastingFilterType;
    CastingFilterType::Pointer caster = CastingFilterType::New();

    caster->SetInput( thresholdConnected->GetOutput() );
    caster->Update();
    trachea = caster->GetOutput();

    /** COMPUTING THE LABEL SIZES */
    cip::LabelMapType::Pointer tracheaAxialCopy = cip::LabelMapType::New();
    cip::LabelMapType::Pointer tracheaCoronalCopy = cip::LabelMapType::New();

    typedef itk::ImageDuplicator<cip::LabelMapType> DuplicatorFilterType;

    DuplicatorFilterType::Pointer duplicatorFilter = DuplicatorFilterType::New();
    duplicatorFilter->SetInputImage(trachea);
    duplicatorFilter->Update();

    // Extracting the axial slice containing the trachea fiducial point
    cip::LabelMapType::SizeType  oneAxialSliceSize;
    cip::CTType::IndexType  indexAxialSlice = indexFiducialSlice;

    oneAxialSliceSize[0] = trachea->GetLargestPossibleRegion().GetSize(0);
    oneAxialSliceSize[1] = trachea->GetLargestPossibleRegion().GetSize(1);
    unsigned int diff = trachea->GetLargestPossibleRegion().GetSize(2)-indexAxialSlice[2];
    if( trachea->GetLargestPossibleRegion().GetSize(2) > 40 &&
        indexAxialSlice[2] >= 20 &&
        diff >= 20 )
    {
        oneAxialSliceSize[2] = 40;
        indexAxialSlice[2]  -= 20;
    }
    else if( trachea->GetLargestPossibleRegion().GetSize(2) > 40 &&
             indexAxialSlice[2] >= 20 &&
             diff < 20 )
    {
        oneAxialSliceSize[2] = 40;
        indexAxialSlice[2]   = trachea->GetLargestPossibleRegion().GetSize(2) - 40;
    }
    else if( trachea->GetLargestPossibleRegion().GetSize(2) > 40 && indexAxialSlice[2] < 20 )
    {
        oneAxialSliceSize[2] = 40;
        indexAxialSlice  [2] = 0;
    }
    else if( trachea->GetLargestPossibleRegion().GetSize(2) <= 40 )
    {
        oneAxialSliceSize[2] = trachea->GetLargestPossibleRegion().GetSize(2);
        indexAxialSlice  [2] = 0;
    }

    cip::LabelMapType::RegionType axialSlice;

    typedef itk::RegionOfInterestImageFilter< cip::LabelMapType, cip::LabelMapType > ROIFilterType;
    ROIFilterType::Pointer axialTracheaFilter = ROIFilterType::New();

    typedef itk::BinaryImageToShapeLabelMapFilter< cip::LabelMapType > ShapeLabelType;
    ShapeLabelType::Pointer axialLabelSizeFilter = ShapeLabelType::New();

    axialLabelSizeFilter->SetInputForegroundValue( cip::AIRWAY );
    axialLabelSizeFilter->SetFullyConnected(1);

    // Extracting the coronal slice containing the trachea fiducial point
    cip::LabelMapType::SizeType oneCoronalSliceSize;
    oneCoronalSliceSize[0] = trachea->GetLargestPossibleRegion().GetSize(0);
    oneCoronalSliceSize[1] = 1;
    oneCoronalSliceSize[2] = 6;

    cip::CTType::IndexType indexCoronalSlice;
    indexCoronalSlice.Fill(0);
    indexCoronalSlice[1] = index[1];
    if( indexFiducialSlice[2] >= 3 )
    {
        indexCoronalSlice[2] = indexFiducialSlice[2] - 3;
    }
    else
    {
        indexCoronalSlice[2] = indexFiducialSlice[2];
    }
    cip::LabelMapType::RegionType coronalSlice;

    ROIFilterType::Pointer coronalTracheaFilter = ROIFilterType::New();

    ShapeLabelType::Pointer coronalLabelSizeFilter = ShapeLabelType::New();
    coronalLabelSizeFilter->SetInputForegroundValue( cip::AIRWAY );
    coronalLabelSizeFilter->SetFullyConnected(1);

    // Computing the sizes 
    double xSize      = 0;
    double ySize      = 0;
    bool   firstCheck = 0;
    bool   check      = 0;
    bool   decrease   = 0;

    double tracheaYSize  = trachea->GetLargestPossibleRegion().GetSize(1) * 0.25;
    double tracheaXSize  = trachea->GetLargestPossibleRegion().GetSize(0) * 2/3;

    do{
        axialSlice.SetSize( oneAxialSliceSize );
        axialSlice.SetIndex( indexAxialSlice );

        duplicatorFilter->Update();
        tracheaAxialCopy = duplicatorFilter->GetOutput();

        axialTracheaFilter->SetInput( tracheaAxialCopy );
        axialTracheaFilter->SetRegionOfInterest( axialSlice );
        axialTracheaFilter->Update();

        axialLabelSizeFilter->SetInput( axialTracheaFilter->GetOutput() );
        axialLabelSizeFilter->Update();

        if( axialLabelSizeFilter->GetOutput()->GetNumberOfLabelObjects() > 0 )
        {
            bool labelOverSize = 0;
            unsigned int numberOfObjects = axialLabelSizeFilter->GetOutput()->GetNumberOfLabelObjects();

            for( unsigned int i = 0; i < numberOfObjects; ++i )
            {
                ySize += axialLabelSizeFilter->GetOutput()->GetNthLabelObject( i )->GetBoundingBox().GetSize(1);
                if( ySize > tracheaYSize)
                {
                    UpperThreshold = UpperThreshold - 20;

                    thresholdConnected->SetUpper( UpperThreshold );
                    caster->SetInput( thresholdConnected->GetOutput() );
                    caster->Update();
                    trachea = caster->GetOutput();
                    duplicatorFilter->Update();
                    tracheaAxialCopy = duplicatorFilter->GetOutput();
                    axialTracheaFilter->SetInput( tracheaAxialCopy );
                    axialTracheaFilter->SetRegionOfInterest( axialSlice );
                    axialTracheaFilter->Update();
                    axialLabelSizeFilter->SetInput( axialTracheaFilter->GetOutput() );
                    axialLabelSizeFilter->Update();
                    decrease = 1;
                    xSize = 0;
                    ySize = 0;
                    labelOverSize = 1;
                    i = numberOfObjects - 1;
                }
            }

            if( !labelOverSize )
            {
                coronalSlice.SetIndex( indexCoronalSlice );
                coronalSlice.SetSize( oneCoronalSliceSize );

                duplicatorFilter->Update();
                tracheaCoronalCopy = duplicatorFilter->GetOutput();

                coronalTracheaFilter->SetInput( tracheaCoronalCopy );
                coronalTracheaFilter->SetRegionOfInterest( coronalSlice );
                coronalTracheaFilter->Update();

                coronalLabelSizeFilter->SetInput( coronalTracheaFilter->GetOutput() );
                coronalLabelSizeFilter->Update();

                unsigned int numberOfObjects = coronalLabelSizeFilter->GetOutput()->GetNumberOfLabelObjects();

                for( unsigned int i = 0; i < numberOfObjects; i++ )
                {
                    xSize += coronalLabelSizeFilter->GetOutput()->GetNthLabelObject( i )->GetBoundingBox().GetSize(0);

                    if( xSize > tracheaXSize )
                    {
                        UpperThreshold = UpperThreshold - 20;

                        thresholdConnected->SetUpper( UpperThreshold );
                        caster->SetInput( thresholdConnected->GetOutput() );
                        caster->Update();
                        trachea = caster->GetOutput();
                        duplicatorFilter->Update();
                        tracheaCoronalCopy = duplicatorFilter->GetOutput();
                        coronalTracheaFilter->SetInput( tracheaCoronalCopy );
                        coronalTracheaFilter->SetRegionOfInterest( coronalSlice );
                        coronalTracheaFilter->Update();
                        i = numberOfObjects - 1;
                        coronalLabelSizeFilter->SetInput( axialTracheaFilter->GetOutput() );
                        coronalLabelSizeFilter->Update();
                        decrease = 1;
                        xSize = 0;
                        ySize = 0;
                    }
                }
            }
            if( xSize != 0 && ySize != 0 )
            {
                xSize = xSize + xSize * 30 / 100;
                ySize = ySize + ySize * 30 / 100;
                firstCheck = 1;
            }
        }
        else
        {
            bool isMinor = 0;
            cip::CTType::SizeType    radius,regionSize;
            cip::CTType::IndexType   regionIndex;
            cip::CTType::RegionType  region;

            regionSize.Fill(3);
            regionIndex = index;
            radius.Fill(3);

            region.SetSize(regionSize);
            region.SetIndex(regionIndex);

            typedef itk::ConstNeighborhoodIterator< cip::CTType > NeighborhoodIterator;
            NeighborhoodIterator iterator(radius, VOI, region);

            unsigned int counter = 0;

            while( counter < iterator.Size() && !isMinor )
            {
                if( iterator.GetPixel(counter) < UpperThreshold )
                {
                    index = iterator.GetIndex( counter );

                    indexCoronalSlice[1] = index[1];
                    indexCoronalSlice[2] = index[2] - 3;

                    thresholdConnected->ClearSeeds();
                    thresholdConnected->AddSeed( index );
                    thresholdConnected->Update();

                    caster->SetInput( thresholdConnected->GetOutput() );
                    caster->Update();

                    trachea = caster->GetOutput();
                    isMinor = 1;
                }
                counter++;
            }

            if ( !isMinor && !decrease)
            {
                if( UpperThreshold < -800 )
                {
                    UpperThreshold = UpperThreshold + 50;

                    thresholdConnected->SetUpper( UpperThreshold );
                    thresholdConnected->Update();

                    caster->SetInput( thresholdConnected->GetOutput() );
                    caster->Update();

                    trachea = caster->GetOutput();
                }
                else
                {
                    std::cout<<"Please move the seed point in a different position."<<std::endl;
                    return trachea;
                }
            }
            else if( !isMinor && decrease )
            {
                if( UpperThreshold < -800 )
                {
                    UpperThreshold = UpperThreshold + 1;

                    thresholdConnected->SetUpper( UpperThreshold );
                    thresholdConnected->Update();

                    caster->SetInput( thresholdConnected->GetOutput() );
                    caster->Update();

                    trachea = caster->GetOutput();
                }
                else
                {
                    std::cout<<"Please move the seed point to a different location."<<std::endl;
                    return trachea;
                }
            }

            axialSlice.SetSize( oneAxialSliceSize );
            axialSlice.SetIndex( indexAxialSlice );

            duplicatorFilter->Update();
            tracheaAxialCopy = duplicatorFilter->GetOutput();

            axialTracheaFilter->SetInput( tracheaAxialCopy );
            axialTracheaFilter->SetRegionOfInterest( axialSlice );
            axialTracheaFilter->Update();

            axialLabelSizeFilter->SetInput( axialTracheaFilter->GetOutput() );
            axialLabelSizeFilter->Update();

            coronalSlice.SetIndex( indexCoronalSlice );
            coronalSlice.SetSize( oneCoronalSliceSize );

            duplicatorFilter->Update();
            tracheaCoronalCopy = duplicatorFilter->GetOutput();

            coronalTracheaFilter->SetInput( tracheaCoronalCopy );
            coronalTracheaFilter->SetRegionOfInterest( coronalSlice );
            coronalTracheaFilter->Update();

            coronalLabelSizeFilter->SetInput( coronalTracheaFilter->GetOutput() );
            coronalLabelSizeFilter->Update();

            xSize = 0;
            ySize = 0;
        }
    }
    while( !firstCheck && UpperThreshold > -1100 );

    duplicatorFilter->SetInputImage(trachea);
    duplicatorFilter->Update();
    tracheaPrev = duplicatorFilter->GetOutput();

    /** INCREASING THE THRESHOLD ITERATIVELY UNTIL LEAKAGE OCCURS */
    typedef itk::SubtractImageFilter< cip::LabelMapType,cip::LabelMapType,cip::LabelMapType > SubtractLabelImageType;
    SubtractLabelImageType::Pointer addedLabel = SubtractLabelImageType::New();

    bool overThreshold = 0;

    ShapeLabelType::Pointer labelSizeFilter = ShapeLabelType::New();

    do{
        addedLabel->SetInput1( trachea );
        addedLabel->SetInput2( tracheaPrev );
        addedLabel->Update();

        labelSizeFilter->SetInput( addedLabel->GetOutput() );
        labelSizeFilter->SetInputForegroundValue( cip::AIRWAY );
        labelSizeFilter->Update();
        unsigned int numberOfObjects = labelSizeFilter->GetOutput()->GetNumberOfLabelObjects();
        double       xSz             = 0;
        double       ySz             = 0;
        double       zSz             = 0;

        if( numberOfObjects > 0 )
        {
            for( unsigned int i = 0; i < numberOfObjects; i++ )
            {
                xSz = labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(0);
                ySz = labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(1);
                zSz = labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(2);
                if( xSz > xSize || ySz > ySize || zSz > trachea->GetLargestPossibleRegion().GetSize(2) / 3 )
                {
                    if( decrease )
                    {
                        UpperThreshold = UpperThreshold - 10;
                        thresholdConnected->SetUpper( UpperThreshold );
                        thresholdConnected->Update();
                        caster->SetInput( thresholdConnected->GetOutput() );
                        caster->Update();
                        trachea = caster->GetOutput();
                    }
                    check = 1;
                    i = labelSizeFilter->GetOutput()->GetNumberOfLabelObjects() - 1;
                }
            }
        }
        if( !check )
        {
            if( UpperThreshold < -800 )
            {
                duplicatorFilter->SetInputImage(trachea);
                duplicatorFilter->Update();
                tracheaPrev = duplicatorFilter->GetOutput();
                if( !decrease )
                {
                    UpperThreshold = UpperThreshold + 50;
                }
                else
                {
                    UpperThreshold = UpperThreshold + 10;
                }
                thresholdConnected->SetUpper( UpperThreshold );
                thresholdConnected->Update();
                caster->SetInput( thresholdConnected->GetOutput() );
                caster->Update();
                trachea = caster->GetOutput();
            }
            else
            {
                check = 1;
                overThreshold = 1;
            }
        }
    }
    while( !check );

    // Decreasing the threshold to find a better segmentation
    if( !overThreshold && !decrease )
    {
        while( check && UpperThreshold > -1100 )
        {
            UpperThreshold = UpperThreshold - 10;

            thresholdConnected->SetUpper( UpperThreshold );
            caster->SetInput( thresholdConnected->GetOutput() );
            trachea = caster->GetOutput();
            caster->Update();

            addedLabel->SetInput1( trachea );
            addedLabel->SetInput2( tracheaPrev );
            addedLabel->Update();

            labelSizeFilter->SetInput( addedLabel->GetOutput() );
            labelSizeFilter->Update();

            unsigned int count = 0;
            unsigned int numberOfObjects = labelSizeFilter->GetOutput()->GetNumberOfLabelObjects();

            if( numberOfObjects > 0 )
            {
                for( unsigned int i = 0; i < numberOfObjects; i++ )
                {
                    if( labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(0) < xSize &&
                        labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(1) < ySize &&
                        labelSizeFilter->GetOutput()->GetNthLabelObject(i)->GetBoundingBox().GetSize(2) < trachea->GetLargestPossibleRegion().GetSize(2) / 3)
                    {
                        count++;
                    }
                }
                if( count == numberOfObjects )
                {
                    check = 0;
                }
            }
            else
            {
                check = 0;
            }
        }
    }

    return trachea;
}

/** FUNCTION FOR FINDING THE CARINA POSITION */
std::vector<cip::LabelMapType::IndexType> CarinaFinder(cip::LabelMapType::Pointer tracheaLabel, unsigned short seedSliceIdx )
{
    int          value;
    int          xDist;
    int          yDist;

    unsigned int yFinalMax = 0;
    unsigned int yFinalMin = 0;

    unsigned int xCarinaPosition = 0;
    unsigned int yCarinaPosition = 0;
    unsigned int carinaIdx = 0;

    cip::LabelMapType::SizeType cropSize = tracheaLabel->GetLargestPossibleRegion().GetSize();

    typedef itk::ImageSliceIteratorWithIndex<cip::LabelMapType> SliceIterator;
    SliceIterator sIt(tracheaLabel, tracheaLabel->GetLargestPossibleRegion());

    sIt.SetFirstDirection(0);
    sIt.SetSecondDirection(1);

    sIt.GoToBegin();

    unsigned int limit = seedSliceIdx - 5;

    while (sIt.GetIndex()[2] < limit)
    {
        unsigned int yMaxPos = 0;
        unsigned int yMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(1);

        unsigned int yMidPos = 0;

        unsigned int yCurrentMax = 0;
        unsigned int yCurrentMin = 0;

        unsigned int prevPos = 0;
        unsigned int prevLine = 0;

        unsigned int prevDiff = 0;

        while (!sIt.IsAtEndOfSlice())
        {
            while (!sIt.IsAtEndOfLine())
            {
                value = sIt.Get();
                if (value == cip::AIRWAY)
                {
                    double d = sIt.GetIndex()[1] - prevLine;
                    yDist = std::abs(d);
                    if (prevLine != 0 && yDist <= 5)
                    {
                        if (sIt.GetIndex()[1] > yMaxPos)
                        {
                            yMaxPos = sIt.GetIndex()[1];
                        }
                        if (sIt.GetIndex()[1] < yMinPos)
                        {
                            yMinPos = sIt.GetIndex()[1];
                        }
                        prevLine = sIt.GetIndex()[1];

                        unsigned int diff = yMaxPos - yMinPos;
                        if (diff > prevDiff)
                        {
                            yCurrentMax = yMaxPos;
                            yCurrentMin = yMinPos;
                        }
                    }
                    else if (prevLine != 0 && yDist >= 5)
                    {
                        prevDiff = yCurrentMax - yCurrentMin;

                        yMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(1);
                        yMaxPos = 0;
                        if (sIt.GetIndex()[1] > yMaxPos)
                        {
                            yMaxPos = sIt.GetIndex()[1];
                        }
                        if (sIt.GetIndex()[1] < yMinPos)
                        {
                            yMinPos = sIt.GetIndex()[1];
                        }
                    }
                    prevLine = sIt.GetIndex()[1];
                }
                ++sIt;
            }
            sIt.NextLine();
        }

        if (yCurrentMax > yCurrentMin)
        {
            yMidPos = yCurrentMin + (yCurrentMax - yCurrentMin) / 2;
        }

        sIt.GoToBeginOfSlice();
        while (!sIt.IsAtEndOfSlice())
        {
            while (!sIt.IsAtEndOfLine())
            {
                value = sIt.Get();
                if (value == cip::AIRWAY)
                {
                    xDist = sIt.GetIndex()[0] - prevPos;
                    if (prevPos != 0 && sIt.GetIndex()[1] == yMidPos
                        && xDist >= 10 && xDist < 20
                        && sIt.GetIndex()[0] > int(cropSize[0] / 3))
                    {
                        carinaIdx = sIt.GetIndex()[2];
                        xCarinaPosition = prevPos + (xDist / 2);
                        yCarinaPosition = yMidPos;
                        yFinalMax = yCurrentMax;
                        yFinalMin = yCurrentMin;
                    }
                    prevPos = sIt.GetIndex()[0];
                }
                ++sIt;
            }
            sIt.NextLine();
        }

        sIt.NextSlice();
    }

    std::vector< cip::LabelMapType::IndexType > carinaVector;

    cip::LabelMapType::IndexType carina;
    carina[0] = xCarinaPosition;
    carina[1] = yCarinaPosition;
    carina[2] = carinaIdx;

    cip::LabelMapType::IndexType yFinal;
    yFinal[0] = yFinalMax;
    yFinal[1] = yFinalMin;
    yFinal[2] = 0;

    carinaVector.push_back(carina);
    carinaVector.push_back(yFinal);

    return carinaVector;
}

/** FUNCTION FOR FINDING THE SEED POINTS FOR RIGHT AND LEFT AIRWAY SEGMENTATION */
std::vector<cip::CTType::IndexType> RLSeedPointsFinder(cip::LabelMapType::Pointer tracheaLabel, cip::LabelMapType::IndexType regionIndex, int xCarinaPos, std::vector<int> yFinal)
{
    int value;
    int yFinalMax = yFinal[0];
    int yFinalMin = yFinal[1];

    typedef itk::RegionOfInterestImageFilter< cip::LabelMapType, cip::LabelMapType > ROIFilterType;
    ROIFilterType::Pointer extractCarinaSliceFilter = ROIFilterType::New();

    cip::CTType::SizeType sliceSize = tracheaLabel->GetLargestPossibleRegion().GetSize();
    sliceSize[2] = 1;
    cip::CTType::IndexType sliceIndex;
    sliceIndex.Fill(0);

    cip::CTType::RegionType sliceRegion;
    sliceRegion.SetSize( sliceSize );
    sliceRegion.SetIndex( sliceIndex );

    extractCarinaSliceFilter->SetInput( tracheaLabel );
    extractCarinaSliceFilter->SetRegionOfInterest( sliceRegion );
    extractCarinaSliceFilter->Update();

    typedef itk::ImageSliceIteratorWithIndex<cip::LabelMapType> SliceIterator;
    SliceIterator singleSliceIt( extractCarinaSliceFilter->GetOutput(), extractCarinaSliceFilter->GetOutput()->GetLargestPossibleRegion() );

    singleSliceIt.SetFirstDirection(0);
    singleSliceIt.SetSecondDirection(1);

    singleSliceIt.GoToBegin();

    unsigned int yRightFiducialMaxPos = 0;
    unsigned int yRightFiducialMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(1);
    unsigned int yRightFiducialPos = 0;

    unsigned int yLeftFiducialMaxPos = 0;
    unsigned int yLeftFiducialMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(1);
    unsigned int yLeftFiducialPos = 0;

    while (!singleSliceIt.IsAtEndOfSlice())
    {
        while (!singleSliceIt.IsAtEndOfLine())
        {
            value = singleSliceIt.Get();
            if (value == cip::AIRWAY)
            {
                if (singleSliceIt.GetIndex()[0] > xCarinaPos)
                {
                    if (singleSliceIt.GetIndex()[1] >= yFinalMin && singleSliceIt.GetIndex()[1] <= yFinalMax)
                    {
                        if (singleSliceIt.GetIndex()[1] > yLeftFiducialMaxPos)
                        {
                            yLeftFiducialMaxPos = singleSliceIt.GetIndex()[1];
                        }
                        if (singleSliceIt.GetIndex()[1] < yLeftFiducialMinPos)
                        {
                            yLeftFiducialMinPos = singleSliceIt.GetIndex()[1];
                        }
                    }
                }
                else if (singleSliceIt.GetIndex()[0] <= xCarinaPos)
                {
                    if (singleSliceIt.GetIndex()[1] >= yFinalMin && singleSliceIt.GetIndex()[1] <= yFinalMax)
                    {
                        if (singleSliceIt.GetIndex()[1] > yRightFiducialMaxPos)
                        {
                            yRightFiducialMaxPos = singleSliceIt.GetIndex()[1];
                        }
                        if (singleSliceIt.GetIndex()[1] < yRightFiducialMinPos)
                        {
                            yRightFiducialMinPos = singleSliceIt.GetIndex()[1];
                        }
                    }
                }
            }
            ++singleSliceIt;
        }
        singleSliceIt.NextLine();
    }

    if (yRightFiducialMaxPos > yRightFiducialMinPos)
    {
        yRightFiducialPos = yRightFiducialMinPos + (yRightFiducialMaxPos - yRightFiducialMinPos) / 2;
    }

    if (yLeftFiducialMaxPos > yLeftFiducialMinPos)
    {
        yLeftFiducialPos = yLeftFiducialMinPos + (yLeftFiducialMaxPos - yLeftFiducialMinPos) / 2;
    }

    unsigned int xRightFiducialMaxPos = 0;
    unsigned int xRightFiducialMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(1);
    unsigned int xRightFiducialPos = 0;

    unsigned int xLeftFiducialMaxPos = 0;
    unsigned int xLeftFiducialMinPos = tracheaLabel->GetLargestPossibleRegion().GetSize(0);
    unsigned int xLeftFiducialPos = 0;

    singleSliceIt.GoToBegin();
    while (!singleSliceIt.IsAtEndOfSlice())
    {
        while (!singleSliceIt.IsAtEndOfLine())
        {
            value = singleSliceIt.Get();
            if (value == cip::AIRWAY)
            {
                if (singleSliceIt.GetIndex()[1] == yRightFiducialPos)
                {
                    if (singleSliceIt.GetIndex()[0] <= xCarinaPos)
                    {
                        if (singleSliceIt.GetIndex()[0] > xRightFiducialMaxPos)
                        {
                            xRightFiducialMaxPos = singleSliceIt.GetIndex()[0];
                        }
                        if (singleSliceIt.GetIndex()[0] < xRightFiducialMinPos)
                        {
                            xRightFiducialMinPos = singleSliceIt.GetIndex()[0];
                        }
                    }
                }
                if (singleSliceIt.GetIndex()[1] == yLeftFiducialPos)
                {
                    if (singleSliceIt.GetIndex()[0] > xCarinaPos)
                    {
                        if (singleSliceIt.GetIndex()[0] > xLeftFiducialMaxPos)
                        {
                            xLeftFiducialMaxPos = singleSliceIt.GetIndex()[0];
                        }
                        if (singleSliceIt.GetIndex()[0] < xLeftFiducialMinPos)
                        {
                            xLeftFiducialMinPos = singleSliceIt.GetIndex()[0];
                        }
                    }
                }
            }
            ++singleSliceIt;
        }
        singleSliceIt.NextLine();
    }

    if (xRightFiducialMaxPos > xRightFiducialMinPos)
    {
        xRightFiducialPos = xRightFiducialMinPos + (xRightFiducialMaxPos - xRightFiducialMinPos) / 2;
    }

    if (xLeftFiducialMaxPos > xLeftFiducialMinPos)
    {
        xLeftFiducialPos = xLeftFiducialMinPos + (xLeftFiducialMaxPos - xLeftFiducialMinPos) / 2;
    }

    cip::CTType::IndexType rightFiducial, leftFiducial;

    rightFiducial[0] = regionIndex[0] + xRightFiducialPos;
    rightFiducial[1] = yRightFiducialPos;
    rightFiducial[2] = regionIndex[2]; //+ trachea->GetLargestPossibleRegion().GetSize(2) + carinaIdx;

    leftFiducial[0] = regionIndex[0] + xLeftFiducialPos;
    leftFiducial[1] = yLeftFiducialPos;
    leftFiducial[2] = rightFiducial[2];

    std::vector<cip::CTType::IndexType> seedPoints;
    seedPoints.push_back(rightFiducial);
    seedPoints.push_back(leftFiducial);

    return seedPoints;
}



/** FUNCTION FOR RIGHT AND LEFT AIRWAYS SEGMENTATION */
cip::LabelMapType::Pointer RightLeftSegmentation( cip::CTType::Pointer VOI,
                                                  cip::CTType::IndexType index,
                                                  std::string reconKernel,
                                                  int trachea_voxels )
{
    /** The method for the segmentation of right and left airways is based on Tschirren (2009):
        Tschirren, J. et al. "Airway segmentation framework for clinical environments." Proc. of Second International Workshop on Pulmonary Image Analysis. 2009.
        As an alternative, Gao's method (2011) may also be used:
        Gao, D. et al. "MGRG-morphological gradient based 3D region growing algorithm for airway tree segmentation in image guided intervention therapy." Bioelectronics and Bioinformatics (ISBB), 2011 International Symposium on. IEEE, 2011.
    */

    double th = 0;
    if( reconKernel == "STANDARD" || reconKernel == "B20f" || reconKernel == "B30f" || reconKernel == "B"
        || reconKernel == "C" || reconKernel == "FC10" || reconKernel == "FC12")
    {
        if( VOI->GetLargestPossibleRegion().GetSize(2) <= 300 )
        {
            if( trachea_voxels > 50000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels > 20000 && trachea_voxels <= 50000 )	//( trachea_voxels > 40000 && trachea_voxels <= 100000 )
            {
                th = 0.75;
            }
            else if( trachea_voxels <= 20000 )
            {
                th = 0.9;
            }
        }
        if( VOI->GetLargestPossibleRegion().GetSize(2) > 300 && VOI->GetLargestPossibleRegion().GetSize(2) <= 400 )
        {
            if( trachea_voxels > 100000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels > 85000 && trachea_voxels <= 100000 )
            {
                th = 0.75;
            }
            else if( trachea_voxels <= 85000 )
            {
                th = 0.9;
            }
        }
        if( VOI->GetLargestPossibleRegion().GetSize(2) > 400 )
        {
            if( trachea_voxels > 170000 )
            {
                //    th = 0.5;
                th = 0.8;
            }
            else if( trachea_voxels > 140000 && trachea_voxels <= 170000 )
            {
                th = 0.75;
            }
            else if( trachea_voxels <= 140000 )
            {
                th = 0.95;
            }
        }
    }
    else if( reconKernel == "LUNG" || reconKernel == "B50f" || reconKernel == "B60f"
             || reconKernel == "D" || reconKernel == "FC50" || reconKernel == "FC52" )
    {
        if( VOI->GetLargestPossibleRegion().GetSize(2) <= 300 )
        {
            if( trachea_voxels > 85000 )
            {
                th = 0.2;
            }
            else if( trachea_voxels > 75000 && trachea_voxels <= 85000 )
            {
                th = 0.3;
            }
            else if( trachea_voxels > 35000 && trachea_voxels <= 75000 )
            {
                th = 0.35;
            }
            else if( trachea_voxels > 10000 && trachea_voxels <= 35000 )//( trachea_voxels > 40000 && trachea_voxels <= 100000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels <= 10000 )
            {
                th = 0.8;
            }
        }
        if( VOI->GetLargestPossibleRegion().GetSize(2) > 300 && VOI->GetLargestPossibleRegion().GetSize(2) <= 400 )
        {
            if( trachea_voxels > 120000 )
            {
                th = 0.2;
            }
            else if( trachea_voxels > 100000 && trachea_voxels <= 120000 )
            {
                th = 0.35;
            }
            else if( trachea_voxels > 85000 && trachea_voxels <= 100000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels <= 85000 )
            {
                th = 0.75;
            }
        }
        if( VOI->GetLargestPossibleRegion().GetSize(2) > 400 )
        {
            if( trachea_voxels > 140000 )
            {
                th = 0.2;
            }
            else if( trachea_voxels > 115000 && trachea_voxels <= 140000 )
            {
                th = 0.35;
            }
            else if( trachea_voxels > 80000 && trachea_voxels <= 115000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels <= 80000 )
            {
                th = 0.75;
            }
        }
    }
    else if( reconKernel == "B70f" || reconKernel == "B70s" )
    {
        if(VOI->GetLargestPossibleRegion().GetSize(2) < 300 )
        {
            if( trachea_voxels > 90000)
            {
                th = 0.35;
            }
            else if( trachea_voxels > 60000 && trachea_voxels <= 90000 )
            {
                th = 0.5;
            }
            else if( trachea_voxels > 30000 && trachea_voxels <= 60000 )
            {
                th = 0.6;
            }
            else if( trachea_voxels <= 30000 )
            {
                th = 0.8;
            }
        }
        if(VOI->GetLargestPossibleRegion().GetSize(2) > 300 )
        {
            if( trachea_voxels > 120000)
            {
                th = 0.25;
            }
            else if( trachea_voxels > 80000 && trachea_voxels <= 120000 )
            {
                th = 0.4;
            }
            else if( trachea_voxels > 50000 && trachea_voxels <= 80000 )
            {
                th = 0.6;
            }
            else if( trachea_voxels <= 50000 )
            {
                th = 0.8;
            }
        }
    }

    double g_max         = 1.6; // 0.15 according to Gao's idea
    double g             = 0;

    double n_voxels      = 0;
    double n_voxels_prev = 1;
    double n_voxels_max  = trachea_voxels * th; // To take into account the fact that half trachea is used to mask the input image

    /*if( n_voxels_max < 20000 )
    {
        n_voxels_max = 20000;
    }*/

    /** SEGMENTATION PIPELINE */

    typedef itk::ConnectedThresholdImageFilter< cip::CTType, cip::CTType > ConnectedFilterType;
    ConnectedFilterType::Pointer thresholdConnected = ConnectedFilterType::New();

    thresholdConnected->SetInput( VOI );
    thresholdConnected->SetReplaceValue( cip::AIRWAY );

    signed short UpperThreshold = -930;

    thresholdConnected->SetUpper( UpperThreshold );
    thresholdConnected->AddSeed( index );

    typedef itk::CastImageFilter<cip::CTType, cip::LabelMapType> CastingFilterType;
    CastingFilterType::Pointer caster = CastingFilterType::New();

    caster->SetInput( thresholdConnected->GetOutput() );
    caster->Update();

    // The number of voxels resulting from the first segmentation are counted
    typedef itk::StatisticsImageFilter< cip::LabelMapType > StatisticsImageFilterType;
    StatisticsImageFilterType::Pointer StatisticsFilter = StatisticsImageFilterType::New();

    StatisticsFilter->SetInput(caster->GetOutput());
    StatisticsFilter->Update();
    n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

    // If the starting threshold gives an empty segmentation, neighbours are checked
    cip::CTType::SizeType radius,regionSize;
    cip::CTType::IndexType regionIndex;
    cip::CTType::RegionType region;

    if( n_voxels == 0 )
    {
        bool isMinor = 0;
        regionSize.Fill(3);
        regionIndex = index;
        radius.Fill(3);

        region.SetSize(regionSize);
        region.SetIndex(regionIndex);

        typedef itk::ConstNeighborhoodIterator< cip::CTType > NeighborhoodIterator;
        NeighborhoodIterator iterator(radius, VOI, region);

        unsigned int counter = 0;

        while( counter < iterator.Size() && !isMinor )
        {
            if( iterator.GetPixel(counter) < UpperThreshold )
            {
                index = iterator.GetIndex( counter );

                thresholdConnected->ClearSeeds();
                thresholdConnected->AddSeed( index );
                thresholdConnected->Update();

                caster->SetInput( thresholdConnected->GetOutput() );
                caster->Update();

                isMinor = 1;
            }
            counter++;
        }
        if ( !isMinor )
        {
            std::cout<<"Please move the seed point in a different position."<<std::endl;
            return caster->GetOutput();
        }

        StatisticsFilter->SetInput(caster->GetOutput());
        StatisticsFilter->Update();
        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );
    }

    // If the number of voxels resulting form the segmentation is too high the threshold is iteratively decreased
    if( n_voxels > n_voxels_max )
    {
        while( n_voxels > n_voxels_max )
        {
            UpperThreshold = UpperThreshold - 10;

            thresholdConnected->SetUpper( UpperThreshold );
            thresholdConnected->Update();
            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();

            StatisticsFilter->SetInput(caster->GetOutput());
            StatisticsFilter->Update();
            n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

            if( n_voxels < 5000 )
            {
                bool isMinor = 0;

                regionIndex = index;

                regionSize.Fill(3);
                radius.Fill(3);
                region.SetSize(regionSize);
                region.SetIndex(regionIndex);

                typedef itk::ConstNeighborhoodIterator< cip::CTType > NeighborhoodIterator;
                NeighborhoodIterator iterator(radius, VOI, region);

                unsigned int counter = 0;

                while( counter < iterator.Size() && !isMinor )
                {
                    if( iterator.GetPixel(counter) < UpperThreshold )
                    {
                        index = iterator.GetIndex( counter );

                        thresholdConnected->ClearSeeds();
                        thresholdConnected->AddSeed( index );
                        thresholdConnected->Update();

                        caster->SetInput( thresholdConnected->GetOutput() );
                        caster->Update();

                        StatisticsFilter->SetInput(caster->GetOutput());
                        StatisticsFilter->Update();
                        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

                        if( n_voxels > 4000 )
                        {
                            isMinor = 1;
                        }
                    }
                    counter++;
                }
            }
        }

        // If n_voxels is too small, an increase of the threshold of even 1 HU might cause the violation of g < g_max
        while( g < g_max && n_voxels < n_voxels_max && UpperThreshold <= -800)
        {
            if( n_voxels > 5000 )
            {
                n_voxels_prev = n_voxels;
            }
            UpperThreshold = UpperThreshold + 1;
            thresholdConnected->SetUpper( UpperThreshold );
            thresholdConnected->Update();
            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();

            StatisticsFilter->SetInput(caster->GetOutput());
            StatisticsFilter->Update();

            n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

            if( n_voxels < 5000 )
            {
                bool isMinor = 0;

                regionIndex = index;
                regionSize.Fill(3);
                radius.Fill(3);

                region.SetSize(regionSize);
                region.SetIndex(regionIndex);

                typedef itk::ConstNeighborhoodIterator< cip::CTType > NeighborhoodIterator;
                NeighborhoodIterator iterator(radius, VOI, region);

                unsigned int counter = 0;

                while( counter < iterator.Size() && !isMinor)
                {
                    if( iterator.GetPixel(counter) < UpperThreshold )
                    {
                        index = iterator.GetIndex( counter );

                        thresholdConnected->ClearSeeds();
                        thresholdConnected->AddSeed( index );
                        thresholdConnected->Update();

                        caster->SetInput( thresholdConnected->GetOutput() );
                        caster->Update();

                        StatisticsFilter->SetInput(caster->GetOutput());
                        StatisticsFilter->Update();
                        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

                        if(n_voxels > 4000)
                        {
                            isMinor = 1;
                        }
                    }
                    counter++;
                }
            }
            if( n_voxels_prev > 5000 )
            {
                g = double( n_voxels/n_voxels_prev );
            }
        }

        UpperThreshold = UpperThreshold - 1;
        thresholdConnected->SetUpper( UpperThreshold );
        caster->SetInput( thresholdConnected->GetOutput() );
        caster->Update();

        StatisticsFilter->SetInput(caster->GetOutput());
        StatisticsFilter->Update();
        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

        if( n_voxels < 4000 )
        {
            UpperThreshold = UpperThreshold + 1;
            thresholdConnected->SetUpper( UpperThreshold );
            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();
        }
    }
    else      // The threshold is iteratively increased until leakage occurs
    {
        // If n_voxels is too small, an increase of the threshold of even 1 HU might cause the violation of g < g_max
        if( n_voxels_max > 5000 )
        {
            while( n_voxels < 5000 )
            {
                UpperThreshold = UpperThreshold + 1;
                thresholdConnected->SetUpper( UpperThreshold );

                caster->SetInput( thresholdConnected->GetOutput() );
                caster->Update();

                StatisticsFilter->SetInput(caster->GetOutput());
                StatisticsFilter->Update();

                n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );
            }
        }
        do{
            UpperThreshold = UpperThreshold + 20;

            n_voxels_prev = n_voxels;

            thresholdConnected->SetUpper( UpperThreshold );

            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();

            StatisticsFilter->SetInput(caster->GetOutput());
            StatisticsFilter->Update();

            n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );
            g = double( n_voxels/n_voxels_prev );	// double((n_voxels - n_voxels_prev)/n_voxels_prev) according to Gao et al.
        }while( g < g_max && n_voxels < n_voxels_max && UpperThreshold <= -800 );

        UpperThreshold = UpperThreshold - 20;
        thresholdConnected->SetUpper( UpperThreshold );

        caster->SetInput( thresholdConnected->GetOutput() );
        caster->Update();

        StatisticsFilter->SetInput(caster->GetOutput());
        StatisticsFilter->Update();
        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

        do{
            UpperThreshold = UpperThreshold + 1;
            n_voxels_prev = n_voxels;

            thresholdConnected->SetUpper( UpperThreshold );

            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();

            StatisticsFilter->SetInput(caster->GetOutput());
            StatisticsFilter->Update();

            n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );
            g = double( n_voxels/n_voxels_prev );	// double((n_voxels - n_voxels_prev)/n_voxels_prev) according to Gao et al.
        }while( g < g_max && n_voxels < n_voxels_max && UpperThreshold <= -800 );

        UpperThreshold = UpperThreshold - 1;

        thresholdConnected->SetUpper( UpperThreshold );
        caster->SetInput( thresholdConnected->GetOutput() );
        caster->Update();

        StatisticsFilter->SetInput(caster->GetOutput());
        StatisticsFilter->Update();
        n_voxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

        if( n_voxels_max > 4000 && n_voxels < 4000 )
        {
            UpperThreshold = UpperThreshold + 1;

            thresholdConnected->SetUpper( UpperThreshold );
            caster->SetInput( thresholdConnected->GetOutput() );
            caster->Update();
        }
    }
    return caster->GetOutput();
}


/** FUNCTION WHICH PASTES AN IMAGE IN A SPECIFIED INDEX OF THE DESTINATION IMAGE */
template <class ImageType>
typename ImageType::Pointer Paste( typename ImageType::Pointer sourceImage, typename ImageType::IndexType index, typename ImageType::Pointer destImage)
{
    typedef itk::PasteImageFilter< ImageType, ImageType > pasteImageFilterType;
    typename pasteImageFilterType::Pointer pasteFilter = pasteImageFilterType::New();

    pasteFilter->SetSourceImage(sourceImage);
    pasteFilter->SetDestinationImage(destImage);
    pasteFilter->SetSourceRegion(sourceImage->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(index);

    try
    {
        pasteFilter->Update();
    }
    catch( itk::ExceptionObject & excep )
    {
        std::cerr << "Exception caught in finalAirways!" << std::endl;
        std::cerr << excep << std::endl;
    }

    return pasteFilter->GetOutput();
}


int main( int argc, char *argv[] )
{

    PARSE_ARGS;

    cip::ChestConventions conventions;

    cip::CTReaderType::Pointer CTReader = cip::CTReaderType::New();
    cip::LabelMapWriterType::Pointer airwayWriter = cip::LabelMapWriterType::New();

    CTReader->SetFileName(inputVolume.c_str());


    std::cout << "Reading CT Volume..." << std::endl;

    try
    {
        CTReader->Update();
    }
    catch ( itk::ExceptionObject & excp )
    {
        std::cerr << "Exception caught while reading CT image: " << std::endl;
        std::cerr << excp << std::endl;
        return cip::NRRDREADFAILURE;
    }

    airwayWriter->SetFileName(airwayLabel.c_str());
    airwayWriter->SetUseCompression(1);

    //   // Take care of not-supine scanned datasets
    //   bool flipIm = (reader->GetOutput()->GetDirection()[0][0] == -1 && reader->GetOutput()->GetDirection()[1][1] == -1);

    //   if( flipIm )
    //   {
    //       typedef itk::FlipImageFilter<cip::CTType> flipImageFilterType;
    //       flipImageFilterType::Pointer flipImageFilter = flipImageFilterType::New();
    //       bool axes[3] = {true,true,false};
    //       flipImageFilter->SetInput(reader->GetOutput());
    //       flipImageFilter->SetFlipAxes(axes);
    //       flipImageFilter->Update();
    //       reader->GraftOutput(flipImageFilter->GetOutput());
    //       reader->Update();
    //   }
    //
    // The different labels will be pasted into finalAirways
    cip::LabelMapType::Pointer finalAirways = cip::LabelMapType::New();

    finalAirways->SetRegions(CTReader->GetOutput()->GetRequestedRegion());
    finalAirways->SetBufferedRegion(CTReader->GetOutput()->GetBufferedRegion());
    finalAirways->SetLargestPossibleRegion(CTReader->GetOutput()->GetLargestPossibleRegion());
    finalAirways->CopyInformation(CTReader->GetOutput());
    finalAirways->Allocate();
    finalAirways->FillBuffer(0);

    cip::CTType::IndexType tracheaFiducial;
    cip::CTType::PointType tracheaPoint;

    if( seed.size() == 1 )
    {
        // Convert to lps the seed point
        tracheaPoint[0] = seed[0][0] * (-CTReader->GetOutput()->GetDirection()[0][0]);
        tracheaPoint[1] = seed[0][1] * (-CTReader->GetOutput()->GetDirection()[1][1]);
        tracheaPoint[2] = seed[0][2] *   CTReader->GetOutput()->GetDirection()[2][2];

        // Convert the lps physical point to index
        CTReader->GetOutput()->TransformPhysicalPointToIndex(tracheaPoint, tracheaFiducial);
    }
    else
    {
        if( seed.size() == 0 )
        {
            std::cerr << "No seeds specified!" << std::endl;
            return cip::ARGUMENTPARSINGERROR;
        }
        else
        {
            std::cerr << "Only one seed point allowed!" << std::endl;
            return cip::ARGUMENTPARSINGERROR;
        }
    }

    /** TRACHEA SEGMENTATION */

    cip::CTType::SizeType  cropSize;
    cip::CTType::IndexType tracheaCropStart;

    cropSize[0] = 70; // Starting value
    tracheaCropStart[0] = tracheaFiducial[0] - 35; // Starting value 
    if( CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) > 200 )
    {
        cropSize[2] = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) - 100;
        tracheaCropStart[2] = 100;
    }
    else
    {
        cropSize[2] = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) - (CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) / 4); // Starting value
        tracheaCropStart[2] = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) / 4; // Starting value
    }

    cropSize[1] = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(1);
    tracheaCropStart[1] = 0;

    cip::CTType::RegionType DesiredRegion;
    DesiredRegion.SetSize(  cropSize  );
    DesiredRegion.SetIndex( tracheaCropStart );

    // Cropping the trachea 
    typedef itk::RegionOfInterestImageFilter< cip::CTType, cip::CTType > inputROIFilterType;
    inputROIFilterType::Pointer ROIFilter = inputROIFilterType::New();

    ROIFilter->SetInput( CTReader->GetOutput() );
    ROIFilter->SetRegionOfInterest( DesiredRegion );
    ROIFilter->Update();

    // Segmenting the trachea
    cip::LabelMapType::Pointer trachea = cip::LabelMapType::New();

    cip::CTType::IndexType FiducialSlice;
    FiducialSlice.Fill(0);
    FiducialSlice[2] = cropSize[2] - ( CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) - tracheaFiducial[2] );

    double fidPs  = double( FiducialSlice[2] );
    double trSz   = double( cropSize[2] );
    double ratio  = fidPs/trSz;

    if(ratio >= 0.85)
    {
        FiducialSlice[2] = cropSize[2]*0.8;
    }

    std::cout << "Segmenting Trachea..." << std::endl;
    trachea = TracheaSegmentation(ROIFilter->GetOutput(), FiducialSlice, seed);

    typedef itk::BinaryBallStructuringElement< cip::LabelMapType::PixelType, DIM > StructuringElementType;

    StructuringElementType structElement;
    StructuringElementType::SizeType radius;
    radius.Fill( 1 );

    structElement.SetRadius( radius );
    structElement.CreateStructuringElement();

    typedef itk::BinaryMorphologicalClosingImageFilter < cip::LabelMapType, cip::LabelMapType, StructuringElementType > CloseType;
    CloseType::Pointer closing = CloseType::New();

    closing->SetInput( trachea );
    closing->SetKernel( structElement );
    closing->SetForegroundValue( cip::AIRWAY );
    closing->Update();
    trachea = closing->GetOutput();

    /*FINDING THE CARINA IN THE TRACHEA */

    std::cout << "Looking for Carina Position..." << std::endl;
    std::vector<cip::LabelMapType::IndexType> carinaVector = CarinaFinder(trachea, FiducialSlice[2]);

    unsigned int xCarinaPosition = carinaVector[0][0];
    unsigned int yCarinaPosition = carinaVector[0][1];
    unsigned int carinaIdx       = carinaVector[0][2];

    cropSize[2] -= carinaIdx;
    tracheaCropStart[2] += carinaIdx;
    FiducialSlice[2] = cropSize[2] - ( CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(2) - tracheaFiducial[2] );

    fidPs  = double( FiducialSlice[2] );
    trSz   = double( cropSize[2] );
    ratio  = fidPs/trSz;

    if(ratio >= 0.85)
    {
        FiducialSlice[2] = cropSize[2]*0.8;
    }

    typedef itk::ImageSliceIteratorWithIndex<cip::LabelMapType> SliceIterator;
    SliceIterator sIt(trachea, trachea->GetLargestPossibleRegion());

    bool exit = 0;
    int value;
    sIt.GoToBegin();

    while( !exit && !sIt.IsAtEnd() )
    {
        while( !exit && !sIt.IsAtEndOfSlice() )
        {
            while( !exit && !sIt.IsAtEndOfLine() )
            {
                value = sIt.Get();
                if( value == cip::AIRWAY && sIt.GetIndex()[2] >= (carinaIdx + 10) && sIt.GetIndex()[2] <= (cropSize[2] - 5) )
                {
                    if( sIt.GetIndex()[0] == 0 || sIt.GetIndex()[0] == cropSize[0] )
                    {
                        cropSize[0] += 10;
                        tracheaCropStart[0] -= 5;
                        xCarinaPosition += 5;
                        exit = 1;
                    }
                }
                ++sIt;
            }
            sIt.NextLine();
        }
        sIt.NextSlice();
    }

    DesiredRegion.SetSize(  cropSize  );
    DesiredRegion.SetIndex( tracheaCropStart );

    ROIFilter->SetInput( CTReader->GetOutput() );
    ROIFilter->SetRegionOfInterest( DesiredRegion );
    ROIFilter->Update();

    trachea = TracheaSegmentation( ROIFilter->GetOutput(), FiducialSlice, seed );

    if( regionSegmentation == "Trachea" )
    {
        /** TRACHEA LABEL CREATION */

        CloseType::Pointer finalClosing = CloseType::New();

        finalClosing->SetInput( trachea );
        finalClosing->SetKernel( structElement );
        finalClosing->SetForegroundValue( cip::AIRWAY );
        finalClosing->SetSafeBorder( 1 );
        finalClosing->Update();

        std::cout << "Writing Trachea LabelMap..." << std::endl;
        airwayWriter->SetInput( finalClosing->GetOutput() );
        try
        {
            airwayWriter->Update();
        }
        catch (itk::ExceptionObject & excp)
        {
            std::cerr << "Exception caught while writing the trachea label: " << std::endl;
            std::cerr << excp << std::endl;
            return cip::EXITFAILURE;
        }
        return EXIT_SUCCESS;
    }

    /** Find right and left airways seed points */

    std::vector<int> yFinalMaxMin;
    yFinalMaxMin.push_back(carinaVector[1][0]);
    yFinalMaxMin.push_back(carinaVector[1][1]);

    std::vector<cip::LabelMapType::IndexType> RLSeeds;

    RLSeeds = RLSeedPointsFinder(trachea, tracheaCropStart, xCarinaPosition, yFinalMaxMin);

    cip::LabelMapType::IndexType rightFiducial = RLSeeds[0];
    cip::LabelMapType::IndexType leftFiducial  = RLSeeds[1];

    /** RIGHT AND LEFT AIRWAYS SEGMENTATION */
    int yDist;

    typedef itk::MaskNegatedImageFilter< cip::CTType, cip::LabelMapType, cip::CTType > MaskNegatedImageType;
    MaskNegatedImageType::Pointer maskNegFilter = MaskNegatedImageType::New();

    typedef itk::StatisticsImageFilter< cip::LabelMapType > StatisticsImageFilterType;
    StatisticsImageFilterType::Pointer StatisticsFilter = StatisticsImageFilterType::New();

    StatisticsFilter->SetInput(trachea);
    StatisticsFilter->Update();
    unsigned int numberOfVoxels = ( StatisticsFilter->GetSum() / cip::AIRWAY );

    // Use half trachea to mask the input image
    typedef itk::BinaryImageToShapeLabelMapFilter< cip::LabelMapType > ShapeLabelType;

    typedef ShapeLabelType::OutputImageType ShapeLabelMapType;
    ShapeLabelMapType::Pointer shapeLabelMap = ShapeLabelMapType::New();

    typedef itk::ImageDuplicator<cip::LabelMapType> DuplicatorFilterType;

    /** RIGHT AIRWAY SEGMENTATION */
    cip::LabelMapType::Pointer rightHalfTrachea = cip::LabelMapType::New();

    DuplicatorFilterType::Pointer rightDuplicatorFilter = DuplicatorFilterType::New();
    rightDuplicatorFilter->SetInputImage(trachea);
    rightDuplicatorFilter->Update();
    rightHalfTrachea = rightDuplicatorFilter->GetOutput();

    SliceIterator rightSIt(rightHalfTrachea, rightHalfTrachea->GetRequestedRegion());

    rightSIt.SetFirstDirection(0);
    rightSIt.SetSecondDirection(1);

    rightSIt.GoToBegin();

    while( !rightSIt.IsAtEnd() )
    {
        unsigned int yMaxPos     = 0;
        unsigned int yMinPos     = rightHalfTrachea->GetLargestPossibleRegion().GetSize(1);

        unsigned int yMidPos     = 0;

        unsigned int yCurrentMax = 0;
        unsigned int yCurrentMin = 0;

        unsigned int prevLine    = 0;

        unsigned int prevDiff    = 0;

        while( !rightSIt.IsAtEndOfSlice() )
        {
            while( !rightSIt.IsAtEndOfLine() )
            {
                value = rightSIt.Get();
                if( value == cip::AIRWAY )
                {
                    double d = rightSIt.GetIndex()[1] - prevLine;
                    yDist = std::abs(d);
                    if( prevLine != 0 && yDist <= 5 )
                    {
                        if( rightSIt.GetIndex()[1] > yMaxPos )
                        {
                            yMaxPos = rightSIt.GetIndex()[1];
                        }
                        if( rightSIt.GetIndex()[1] < yMinPos )
                        {
                            yMinPos = rightSIt.GetIndex()[1];
                        }
                        prevLine = rightSIt.GetIndex()[1];

                        unsigned int diff = yMaxPos - yMinPos;
                        if( diff > prevDiff)
                        {
                            yCurrentMax = yMaxPos;
                            yCurrentMin = yMinPos;
                        }
                    }
                    else if( prevLine != 0 && yDist >= 5 )
                    {
                        prevDiff = yCurrentMax - yCurrentMin;

                        yMinPos = rightHalfTrachea->GetLargestPossibleRegion().GetSize(1);
                        yMaxPos = 0;
                        if( rightSIt.GetIndex()[1] > yMaxPos )
                        {
                            yMaxPos = rightSIt.GetIndex()[1];
                        }
                        if( rightSIt.GetIndex()[1] < yMinPos )
                        {
                            yMinPos = rightSIt.GetIndex()[1];
                        }
                    }
                    prevLine = rightSIt.GetIndex()[1];
                }
                ++rightSIt;
            }
            rightSIt.NextLine();
        }

        if( yCurrentMax > yCurrentMin )
        {
            yMidPos = yCurrentMin + (yCurrentMax - yCurrentMin) / 2;
        }

        unsigned int xMaxPos  = 0;
        unsigned int xMinPos  = rightHalfTrachea->GetLargestPossibleRegion().GetSize(0);
        unsigned int xMidPos  = 0;

        rightSIt.GoToBeginOfSlice();
        while( !rightSIt.IsAtEndOfSlice() )
        {
            while( !rightSIt.IsAtEndOfLine() )
            {
                value = rightSIt.Get();
                if( rightSIt.GetIndex()[1] == yMidPos && value == cip::AIRWAY )
                {
                    if( rightSIt.GetIndex()[0] > xMaxPos )
                    {
                        xMaxPos = rightSIt.GetIndex()[0];
                    }
                    if( rightSIt.GetIndex()[0] < xMinPos )
                    {
                        xMinPos = rightSIt.GetIndex()[0];
                    }
                }
                ++rightSIt;
            }
            rightSIt.NextLine();
        }
        if( xMaxPos > xMinPos )
        {
            xMidPos = xMinPos + (xMaxPos - xMinPos) / 2;
        }
        rightSIt.GoToBeginOfSlice();
        while( !rightSIt.IsAtEndOfSlice() )
        {
            while( !rightSIt.IsAtEndOfLine() )
            {
                value = rightSIt.Get();
                if( value == cip::AIRWAY )
                {
                    if (rightSIt.GetIndex()[0] <= xMidPos) // !flipIm && rightSIt.GetIndex()[0] <= xMidPos
                    {
                        rightSIt.Set(0);
                    }
                    //else if( flipIm && rightSIt.GetIndex()[0] >= xMidPos )
                    //{
                    //    rightSIt.Set(0);            
                    //}
                }
                ++rightSIt;
            }
            rightSIt.NextLine();
        }
        rightSIt.NextSlice();
    }

    rightSIt.GoToBegin();
    while( !rightSIt.IsAtEndOfSlice() )
    {
        while( !rightSIt.IsAtEndOfLine() )
        {
            value = rightSIt.Get();
            if( value == cip::AIRWAY )
            {
                rightSIt.Set(0);
            }
            ++rightSIt;
        }
        rightSIt.NextLine();
    }

    cip::CTType::IndexType regionIndex = tracheaCropStart;

    cip::LabelMapType::IndexType idx;
    idx = regionIndex;

    finalAirways->FillBuffer(0);
    finalAirways = Paste<cip::LabelMapType>( rightHalfTrachea, idx, finalAirways );

    cip::LabelMapType::Pointer closeImage = cip::LabelMapType::New();

    closeImage->SetRegions( CTReader->GetOutput()->GetRequestedRegion() );
    closeImage->SetBufferedRegion( CTReader->GetOutput()->GetBufferedRegion());
    closeImage->SetLargestPossibleRegion( CTReader->GetOutput()->GetLargestPossibleRegion());
    closeImage->CopyInformation( CTReader->GetOutput());
    closeImage->Allocate();
    closeImage->FillBuffer( cip::AIRWAY );

    cip::LabelMapType::SizeType sz = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize();

    //   if( !flipIm )
    //   {
    sz[0] = CTReader->GetOutput()->GetLargestPossibleRegion().GetSize(0) - (tracheaCropStart[0] + xCarinaPosition ) + 2;
    //   }
    //   else
    //   {
    //       sz[0] = tracheaCropStart[0] + xCarinaPosition + 2;
    //   }

    sz[2] = 6;
    cip::LabelMapType::IndexType stIdx;
    stIdx.Fill(0);

    cip::LabelMapType::RegionType rg;
    rg.SetSize(  sz  );
    rg.SetIndex( stIdx );

    typedef itk::RegionOfInterestImageFilter< cip::LabelMapType, cip::LabelMapType > outputROIFilterType;
    outputROIFilterType::Pointer cropCloseImageFilter = outputROIFilterType::New();

    cropCloseImageFilter->SetInput( closeImage );
    cropCloseImageFilter->SetRegionOfInterest( rg );
    cropCloseImageFilter->Update();

    //if(!flipIm)
    //{
    stIdx[0] = tracheaCropStart[0] + xCarinaPosition - 2;
    stIdx[2] = rightFiducial[2] - 3;
    //}
    //else
    //{
    //    stIdx[0] = 0;
    //    stIdx[2] = leftFiducial[2] - 3;
    //}	

    finalAirways = Paste<cip::LabelMapType>( cropCloseImageFilter->GetOutput(), stIdx, finalAirways );

    // The half trachea label is used to mask the input image
    maskNegFilter->SetInput( CTReader->GetOutput() );
    maskNegFilter->SetMaskImage( finalAirways );
    maskNegFilter->Update();

    cip::LabelMapType::Pointer rightAirway = cip::LabelMapType::New();
    std::string reconstructionKernel = reconstructionKernelType.c_str();
    //
    //   if( flipIm )
    //   {
    //   rightAirway = RightLeftSegmentation( maskNegFilter->GetOutput(), leftFiducial, reconstructionKernel, numberOfVoxels );
    //   }
    //   else
    //   {
    std::cout << "Segmenting Right Airways..." << std::endl;
    rightAirway = RightLeftSegmentation( maskNegFilter->GetOutput(), rightFiducial, reconstructionKernel, numberOfVoxels);
    //   }

    ShapeLabelType::Pointer rightLabelConverter = ShapeLabelType::New();

    rightLabelConverter->SetInput( rightAirway );
    rightLabelConverter->SetInputForegroundValue( cip::AIRWAY );
    rightLabelConverter->Update();

    shapeLabelMap = rightLabelConverter->GetOutput();

    typedef itk::MergeLabelMapFilter< ShapeLabelMapType > MergeFilterType;
    MergeFilterType::Pointer mergeFilter = MergeFilterType::New();
    mergeFilter->SetMethod(MergeFilterType::PACK);

    ShapeLabelType::Pointer labelConverter = ShapeLabelType::New();
    labelConverter->SetInputForegroundValue(cip::AIRWAY);

    typedef itk::LabelMapToBinaryImageFilter< ShapeLabelMapType, cip::LabelMapType > LabelMapToBinaryImageType;
    LabelMapToBinaryImageType::Pointer labelToBinaryFilter = LabelMapToBinaryImageType::New();

    labelToBinaryFilter->SetBackgroundValue(0);
    labelToBinaryFilter->SetForegroundValue(cip::AIRWAY);

    if (regionSegmentation == "RightAirway")
    {
        finalAirways->FillBuffer(0);
        finalAirways = Paste<cip::LabelMapType>(trachea, regionIndex, finalAirways);

        labelConverter->SetInput( finalAirways );
        labelConverter->Update();

        mergeFilter->SetInput( shapeLabelMap );
        mergeFilter->SetInput( 1, labelConverter->GetOutput() );
        mergeFilter->Update();

        labelToBinaryFilter->SetInput( mergeFilter->GetOutput() );

        labelToBinaryFilter->Update();

        CloseType::Pointer finalClosing = CloseType::New();

        finalClosing->SetInput( labelToBinaryFilter->GetOutput() );
        finalClosing->SetKernel( structElement );
        finalClosing->SetForegroundValue( cip::AIRWAY );
        finalClosing->SetSafeBorder( 1 );
        finalClosing->Update();


        /** RIGHT LABEL CREATION */
        std::cout << "Writing Right Airway LabelMap..." << std::endl;
        airwayWriter->SetInput( finalClosing->GetOutput());
        try
        {
            airwayWriter->Update();
        }
        catch (itk::ExceptionObject & excp)
        {
            std::cerr << "Exception caught while writing the right airway label: " << std::endl;
            std::cerr << excp << std::endl;
            return cip::EXITFAILURE;
        }
        return EXIT_SUCCESS;
    }

    /** LEFT AIRWAY SEGMENTATION */
    cip::LabelMapType::Pointer leftHalfTrachea = cip::LabelMapType::New();

    DuplicatorFilterType::Pointer leftDuplicatorFilter = DuplicatorFilterType::New();
    leftDuplicatorFilter->SetInputImage(trachea);
    leftDuplicatorFilter->Update();
    leftHalfTrachea = leftDuplicatorFilter->GetOutput();

    SliceIterator leftSIt(leftHalfTrachea, leftHalfTrachea->GetRequestedRegion());

    leftSIt.SetFirstDirection(0);
    leftSIt.SetSecondDirection(1);

    leftSIt.GoToBegin();

    while( !leftSIt.IsAtEndOfSlice() )
    {
        while( !leftSIt.IsAtEndOfLine() )
        {
            value = leftSIt.Get();
            if( value == cip::AIRWAY )
            {
                leftSIt.Set(0);
            }
            ++leftSIt;
        }
        leftSIt.NextLine();
    }

    leftSIt.GoToBegin();
    while( !leftSIt.IsAtEnd() )
    {
        unsigned int yMaxPos     = 0;
        unsigned int yMinPos     = leftHalfTrachea->GetLargestPossibleRegion().GetSize(1);

        unsigned int yMidPos     = 0;

        unsigned int yCurrentMax = 0;
        unsigned int yCurrentMin = 0;

        unsigned int prevLine    = 0;

        unsigned int prevDiff    = 0;

        while( !leftSIt.IsAtEndOfSlice() )
        {
            while( !leftSIt.IsAtEndOfLine() )
            {
                value = leftSIt.Get();
                if( value == cip::AIRWAY )
                {
                    double d = leftSIt.GetIndex()[1] - prevLine;
                    yDist = std::abs(d);
                    if( prevLine != 0 && yDist <= 5 )
                    {
                        if( leftSIt.GetIndex()[1] > yMaxPos )
                        {
                            yMaxPos = leftSIt.GetIndex()[1];
                        }
                        if( leftSIt.GetIndex()[1] < yMinPos )
                        {
                            yMinPos = leftSIt.GetIndex()[1];
                        }
                        prevLine = leftSIt.GetIndex()[1];

                        unsigned int diff = yMaxPos - yMinPos;
                        if( diff > prevDiff)
                        {
                            yCurrentMax = yMaxPos;
                            yCurrentMin = yMinPos;
                        }
                    }
                    else if( prevLine != 0 && yDist >= 5 )
                    {
                        prevDiff = yCurrentMax - yCurrentMin;

                        yMinPos = leftHalfTrachea->GetLargestPossibleRegion().GetSize(1);
                        yMaxPos = 0;
                        if( leftSIt.GetIndex()[1] > yMaxPos )
                        {
                            yMaxPos = leftSIt.GetIndex()[1];
                        }
                        if( leftSIt.GetIndex()[1] < yMinPos )
                        {
                            yMinPos = leftSIt.GetIndex()[1];
                        }
                    }
                    prevLine = leftSIt.GetIndex()[1];
                }
                ++leftSIt;
            }
            leftSIt.NextLine();
        }

        if( yCurrentMax > yCurrentMin )
        {
            yMidPos = yCurrentMin + (yCurrentMax - yCurrentMin) / 2;
        }

        unsigned int xMaxPos  = 0;
        unsigned int xMinPos  = leftHalfTrachea->GetLargestPossibleRegion().GetSize(0);
        unsigned int xMidPos  = 0;

        leftSIt.GoToBeginOfSlice();
        while( !leftSIt.IsAtEndOfSlice() )
        {
            while( !leftSIt.IsAtEndOfLine() )
            {
                value = leftSIt.Get();
                if( leftSIt.GetIndex()[1] == yMidPos && value == cip::AIRWAY )
                {
                    if( leftSIt.GetIndex()[0] > xMaxPos )
                    {
                        xMaxPos = leftSIt.GetIndex()[0];
                    }
                    if( leftSIt.GetIndex()[0] < xMinPos )
                    {
                        xMinPos = leftSIt.GetIndex()[0];
                    }
                }
                ++leftSIt;
            }
            leftSIt.NextLine();
        }
        if( xMaxPos > xMinPos )
        {
            xMidPos = xMinPos + (xMaxPos - xMinPos) / 2;
        }

        leftSIt.GoToBeginOfSlice();
        while( !leftSIt.IsAtEndOfSlice() )
        {
            while( !leftSIt.IsAtEndOfLine() )
            {
                value = leftSIt.Get();
                if( value == cip::AIRWAY )
                {
                    if( leftSIt.GetIndex()[0] >= xMidPos ) //!flipIm && leftSIt.GetIndex()[0] >= xMidPos
                    {
                        leftSIt.Set(0);
                    }
                    //                   else if( flipIm && leftSIt.GetIndex()[0] <= xMidPos )
                    //                   {
                    //                       leftSIt.Set(0);
                    //                   }
                }
                ++leftSIt;
            }
            leftSIt.NextLine();
        }
        leftSIt.NextSlice();
    }

    finalAirways->FillBuffer(0);
    finalAirways = Paste<cip::LabelMapType>( leftHalfTrachea, idx, finalAirways );

    //   if( !flipIm )
    //   {
    sz[0] = tracheaCropStart[0] + xCarinaPosition + 2;
    //   }
    //   else
    //   {
    //       sz[0] = reader->GetOutput()->GetLargestPossibleRegion().GetSize(0) - (tracheaCropStart[0] + xCarinaPosition ) + 2;
    //   }
    //
    stIdx.Fill(0);

    rg.SetSize(  sz  );
    rg.SetIndex( stIdx );

    cropCloseImageFilter->SetInput( closeImage );
    cropCloseImageFilter->SetRegionOfInterest( rg );
    cropCloseImageFilter->Update();

    //   if(!flipIm)
    //   {
    stIdx[0] = 0;
    stIdx[2] = leftFiducial[2] - 3;
    //   }
    //   else
    //   {
    //       stIdx[0] = tracheaCropStart[0] + xCarinaPosition - 2;
    //       stIdx[2] = rightFiducial[2] - 3;
    //   }
    finalAirways = Paste<cip::LabelMapType>( cropCloseImageFilter->GetOutput(), stIdx, finalAirways );

    maskNegFilter->SetInput( CTReader->GetOutput() );
    maskNegFilter->SetMaskImage( finalAirways );
    maskNegFilter->Update();

    cip::LabelMapType::Pointer leftAirway = cip::LabelMapType::New();
    //
    //   if( flipIm )
    //   {
    //       leftAirway = RightLeftSegmentation( maskNegFilter->GetOutput(), rightFiducial, reconstructionKernel, numberOfVoxels, cip::AIRWAY );
    //   }
    //   else
    //   {
    std::cout << "Segmenting Left Airways..." << std::endl;
    leftAirway = RightLeftSegmentation(maskNegFilter->GetOutput(), leftFiducial, reconstructionKernel, numberOfVoxels);
    //   }
    //

    ShapeLabelType::Pointer leftLabelConverter = ShapeLabelType::New();

    leftLabelConverter->SetInput( leftAirway );
    leftLabelConverter->SetInputForegroundValue( cip::AIRWAY );
    leftLabelConverter->Update();

    if (regionSegmentation == "LeftAirway")
    {
        finalAirways->FillBuffer(0);
        finalAirways = Paste<cip::LabelMapType>( trachea, regionIndex, finalAirways );

        labelConverter->SetInput( finalAirways );
        labelConverter->Update();

        mergeFilter->SetInput( leftLabelConverter->GetOutput() );
        mergeFilter->SetInput( 1, labelConverter->GetOutput() );
        mergeFilter->Update();

        labelToBinaryFilter->SetInput( mergeFilter->GetOutput() );
        labelToBinaryFilter->Update();

        CloseType::Pointer finalClosing = CloseType::New();

        finalClosing->SetInput( labelToBinaryFilter->GetOutput() );
        finalClosing->SetKernel( structElement );
        finalClosing->SetForegroundValue( cip::AIRWAY );
        finalClosing->SetSafeBorder( 1 );
        finalClosing->Update();

        /** LEFT LABEL CREATION */
        std::cout << "Writing Left Airway LabelMap..." << std::endl;
        airwayWriter->SetInput( finalClosing->GetOutput() );
        try
        {
            airwayWriter->Update();
        }
        catch (itk::ExceptionObject & excp)
        {
            std::cerr << "Exception caught while writing the left airway label: " << std::endl;
            std::cerr << excp << std::endl;
            return cip::EXITFAILURE;
        }
        return EXIT_SUCCESS;
    }

    mergeFilter->SetInput( shapeLabelMap );
    mergeFilter->SetInput( 1, leftLabelConverter->GetOutput() );
    mergeFilter->Update();
    shapeLabelMap = mergeFilter->GetOutput();

    finalAirways->FillBuffer(0);
    finalAirways = Paste<cip::LabelMapType>( trachea, regionIndex, finalAirways );

    labelConverter->SetInput( finalAirways );
    labelConverter->Update();

    for( unsigned int i = 0; i < labelConverter->GetOutput()->GetNumberOfLabelObjects(); i++ )
    {
        shapeLabelMap->PushLabelObject(labelConverter->GetOutput()->GetNthLabelObject(i));
    }

    shapeLabelMap->Update();

    labelToBinaryFilter->SetInput( shapeLabelMap );
    labelToBinaryFilter->Update();

    finalAirways = labelToBinaryFilter->GetOutput();

    ////////////////////////////////////////////////////////
    /*regionIndex.Fill(0);
    finalAirways->FillBuffer(0);
    finalAirways = Paste<cip::LabelMapType>( leftAirway,regionIndex, finalAirways );*/
    ////////////////////////////////////////////////////////

    //   if( flipIm )
    //   {
    //       typedef itk::FlipImageFilter<cip::LabelMapType> flipImageFilterType;
    //       flipImageFilterType::Pointer flipImageFilter = flipImageFilterType::New();
    //       bool axes[3] = {true,true,false};
    //       flipImageFilter->SetInput(finalAirways);
    //       flipImageFilter->SetFlipAxes(axes);
    //       flipImageFilter->Update();
    //       finalAirways = flipImageFilter->GetOutput();
    //   }

    /** CLOSING AND HOLE FILLING TO IMPROVE THE SEGMENTATION RESULT */
    //typedef itk::BinaryBallStructuringElement< cip::LabelMapType::PixelType, DIM > StructuringElementType;
    std::cout << "Final Closing..." << std::endl;
    StructuringElementType newStructElement;
    StructuringElementType::SizeType newRadius;
    newRadius.Fill( 5 );

    newStructElement.SetRadius( newRadius );
    newStructElement.CreateStructuringElement();

    //typedef itk::BinaryMorphologicalClosingImageFilter < cip::LabelMapType, cip::LabelMapType, StructuringElementType > CloseType;
    CloseType::Pointer newClosing = CloseType::New();

    newClosing->SetInput( finalAirways );
    newClosing->SetKernel( newStructElement );
    newClosing->SetForegroundValue( cip::AIRWAY );
    newClosing->SetSafeBorder( 1 );
    newClosing->Update();

    typedef itk::VotingBinaryIterativeHoleFillingImageFilter< cip::LabelMapType > IterativeFillHolesFilterType;
    IterativeFillHolesFilterType::Pointer HoleFilling = IterativeFillHolesFilterType::New();

    cip::LabelMapType::SizeType FillRadius;

    FillRadius.Fill(1);

    HoleFilling->SetInput( newClosing->GetOutput() );
    HoleFilling->SetRadius( FillRadius );
    HoleFilling->SetBackgroundValue( 0 );
    HoleFilling->SetForegroundValue( cip::AIRWAY );
    HoleFilling->SetMajorityThreshold( 1 );
    HoleFilling->SetMaximumNumberOfIterations( 5 );
    HoleFilling->Update();

    typedef itk::GrayscaleFillholeImageFilter< cip::LabelMapType, cip::LabelMapType > GSFillHolesFilterType;
    GSFillHolesFilterType::Pointer GSHoleFilling = GSFillHolesFilterType::New();

    GSHoleFilling->SetInput( HoleFilling->GetOutput() );
    GSHoleFilling->SetFullyConnected( 1 );
    GSHoleFilling->Update();

    /** LABEL CREATION */
    airwayWriter->SetInput( GSHoleFilling->GetOutput() );
    std::cout << "Writing Airway LabelMap..." << std::endl;
    try
    {
        airwayWriter->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
        std::cerr << "Exception caught while writing the airway label: " << std::endl;
        std::cerr << excp << std::endl;
        return cip::LABELMAPWRITEFAILURE;
    }

    std::cout << "DONE." << std::endl;

    return cip::EXITSUCCESS;
}

#endif