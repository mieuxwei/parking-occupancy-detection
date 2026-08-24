# Dataset Research

Research completed for Milestone 2 on 2026-08-21. No dataset files were
downloaded during that milestone. Milestone 3 later acquired only the official
metadata CSV; image archives remain deferred.

## Evaluation Criteria

- Binary occupancy labels suitable for `EMPTY` / `OCCUPIED` classification
- Availability of cropped parking-space images
- Parking lot, camera, date, time, weather, and slot metadata
- Support for leakage-aware group or domain splits
- Download size and acquisition complexity
- Clear license and citation requirements

## Dataset Comparison

| Criterion | PKLot | CNRPark+EXT |
| --- | --- | --- |
| Task labels | Vacant / occupied | Free / busy (`0` / `1` in provided label lists) |
| Parking sites | Two parking lots | One parking lot with multiple cameras |
| Views / cameras | Three views: UFPR04, UFPR05, PUCPR | CNRPark: 2 cameras; CNR-EXT: 9 cameras |
| Full frames | 12,417 images at 1280×720 | 242 CNRPark frames and 4,081 CNR-EXT frames |
| Parking-space samples | 695,899 segmented spaces can be produced from XML annotations | 12,584 CNRPark patches and 144,965 CNR-EXT patches |
| Weather | Sunny, overcast/cloudy, rainy | CNR-EXT: sunny, overcast, rainy |
| Organization / metadata | View, weather, acquisition date, XML polygons and occupancy labels | Camera, class or label, capture date/time, weather, slot ID, and a metadata CSV |
| Cropped patches | Included as `PKLotSegmented` in the official archive, alongside full frames and XML | Official 150×150 patch archives are available |
| Official download size | 4.6 GB archive | 18.1 MB metadata CSV; 36.6 MB CNRPark patches; 449.5 MB CNR-EXT patches |
| License | Creative Commons Attribution 4.0 | Open Data Commons ODbL 1.0 |

The CNRPark and CNR-EXT row counts were later confirmed against the official
metadata CSV. Archive sizes remain the values reported by the maintainers.

## PKLot Assessment

PKLot contains images from the UFPR and PUCPR parking lots in Curitiba, Brazil.
UFPR04 and UFPR05 are two views of the same UFPR parking lot, while PUCPR is a
different parking lot. Images were captured at five-minute intervals and are
organized by view, weather, and acquisition date. Each full image has an XML
annotation containing parking-space polygons and occupied/vacant labels.

### Strengths

- Two physical parking lots make it suitable for a true cross-location test.
- Weather, date, view, and original-frame relationships support group-aware
  experimental design.
- The class distribution is close to balanced overall according to the official
  dataset README.
- CC BY 4.0 permits reuse with attribution and the required license notice.

### Limitations and Risks

- The official archive is 4.9 GB in exact byte units and expands to hundreds of
  thousands of small files. On the project's exFAT SSD it occupies about 91 GiB
  because of filesystem allocation units.
- UFPR04 and UFPR05 are different views of the same site, not independent parking
  locations.
- Five-minute capture intervals create highly similar temporal samples. Patches
  from the same frame or day must not be split across train and test sets.
- The official description states that night images are not available.

## CNRPark+EXT Assessment

CNRPark+EXT was collected at the CNR research-area parking lot in Pisa, Italy.
The preliminary CNRPark subset contains 12,584 patches from 242 frames and two
cameras. CNR-EXT contains 144,965 patches from 4,081 frames acquired on 23 days
by nine cameras. The official site provides ready-made 150×150 parking-space
patches and a separate metadata CSV.

The CNR-EXT path convention exposes weather, capture date, camera, capture time,
and a global slot ID. Provided label lists use `0` for free and `1` for busy.

### Strengths

- Ready-made patches directly match the initial binary classification task.
- The download is substantially smaller than PKLot.
- Camera, date/time, weather, and slot identifiers support leakage audits and
  camera-domain experiments.
- Nine camera perspectives, changing light, shadows, and partial occlusions offer
  useful visual variation.

### Limitations and Risks

- All CNR subsets represent one physical parking area, so holding out a camera is
  a camera-domain test rather than a true cross-location test.
- Camera fields of view can overlap. A split based only on individual patch names
  could leak the same capture event across sets.
- The maintainers provide experiment splits, but they must be audited against this
  project's stricter frame-, time-, and domain-grouping rules before use.
- ODbL attribution and share-alike requirements must be reviewed before any
  redistribution of a derived database.

## Dataset Selection

**Primary dataset: CNRPark+EXT.**

It is the smallest practical starting point, supplies official cropped patches,
and exposes the metadata needed to build a reproducible leakage-aware pipeline.
It will support the initial in-domain and held-out-camera experiments.

**Secondary external domain: PKLot.**

PKLot will be acquired only after the primary data pipeline and leakage checks are
working. Its PUCPR and UFPR sites provide a stronger cross-location and
cross-dataset test than camera holdout within CNRPark+EXT.

This selection does not treat a held-out CNR camera as equivalent to a held-out
parking site; results must label those settings separately.

## Acquisition Strategy

Metadata acquisition began in Milestone 3. The two patch archives were acquired
and extracted to external storage during Milestone 4:

1. Keep the acquired CNRPark+EXT metadata CSV in repository
   `data/raw/cnrpark_ext/`; keep both 150×150 patch archives and extracted images
   under `$PARKING_DATA_ROOT/cnrpark_ext/`.
2. Do not initially download the 1.1 GB full-frame archive, supplied trained
   models, or published split bundles; they are not required for the first
   classification pipeline.
3. Record the source URL, download date, archive size, license, and a locally
   computed SHA-256 checksum in a data manifest.
4. Keep large archives and images under the external `PARKING_DATA_ROOT`; never
   place them in repository `data/` or commit them to Git.
5. Acquire the official 4.6 GB PKLot archive later, when the cross-location stage
   is ready. Preserve its original image/XML structure before deriving patches.

## Split Requirements for Milestone 3

- Keep every patch derived from the same source frame in one split.
- For CNR-EXT, consider grouping simultaneous timestamps across cameras because
  camera views can overlap.
- Separate held-out-camera evaluation from held-out-date evaluation.
- For PKLot, keep each acquisition date in only one split, following the official
  protocol's leakage precaution.
- Treat UFPR04 and UFPR05 as two views of one site; use PUCPR when a distinct
  location is required.
- Do not finalize split ratios until actual metadata has been audited.

## Official Sources

- [PKLot official dataset page](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/)
- [PKLot official dataset README](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
- [PKLot paper](https://doi.org/10.1016/j.eswa.2015.02.009)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [CNRPark+EXT official dataset page](https://cnrpark.it/)
- [CNRPark+EXT paper](https://doi.org/10.1016/j.eswa.2016.10.075)
- [Open Data Commons ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
