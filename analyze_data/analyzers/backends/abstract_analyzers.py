"""
analyze_data/analyzers/backends/abstract_analyzers.py
Pure abstract contracts for all data analysis classes.
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar , Optional

T = TypeVar("T")

class AbstractBaseDataAnalysis(ABC, Generic[T]):
    def __init__(self, data_frame: Optional[T] = None):   
        self._data_frame = data_frame

    def set_data_frame(self, data_frame: T) -> None:
        """Permite inyectar el DataFrame después de la creación."""
        self._data_frame = data_frame

    @abstractmethod
    def analyze(self, **kwargs) -> Any:
        pass

class AbstractAnalyseDataTypes(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataShape(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataInfo(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataDescribe(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataColumns(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataIndex(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataHead(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataTail(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDataSample(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseSeasonality(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseTrendPatterns(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDistributionType(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseSkewnessKurtosis(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseNormalityTests(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseValueCounts(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalysePercentiles(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseFrequencyDistribution(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCentralTendency(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDispersion(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseHypothesisTest(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseAnova(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseChiSquare(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCorrelationSignificance(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseConfidenceIntervals(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseEffectSize(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalysePowerAnalysis(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseBootstrap(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCorrelationMatrix(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseMulticollinearity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseMutualInformation(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalysePartialCorrelation(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCrossCorrelation(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGrangerCausality(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseContingency(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseInteractionEffects(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseFeatureVariance(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseFeatureSelection(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseFeatureImportance(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDimensionalityReduction(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseClassImbalance(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseModelResiduals(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseLearningCurve(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCrossValidation(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseVolatility(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseMomentum(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseMovingAverages(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseStationarity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseLagFeatures(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseChangePoints(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseForecastAccuracy(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCyclicalPatterns(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseRollingStatistics(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseTextBasicStats(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseWordFrequency(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseSentiment(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseTopicDetection(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseLanguageDetection(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseTextSimilarity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseNamedEntityDensity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseKMeansClusters(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseRFMSegmentation(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCohortAnalysis(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalysePopulationSplits(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseDBSCANClusters(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseHierarchicalClusters(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGrowthRates(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseRiskMetrics(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseFinancialRatios(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseConversionFunnel(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseChurnRate(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCustomerLifetimeValue(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseParetoAnalysis(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseRunRate(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGeoDistribution(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGeoClustering(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGeoBoundingBox(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseGeoHeatmap(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseProximity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseNetworkDensity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCentrality(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseCommunityDetection(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalysePathAnalysis(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseKaplanMeier(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseHazardRate(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseEventDensity(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...

class AbstractAnalyseTimeToEvent(AbstractBaseDataAnalysis[T], Generic[T]):
    @abstractmethod
    def analyze(self, **kwargs) -> Any: ...
