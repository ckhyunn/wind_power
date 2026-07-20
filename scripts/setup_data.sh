#!/bin/bash
# ==========================================================
# 데이콘 풍력발전량 예측 대회 - 데이터 세팅 스크립트
#
# 하는 일:
#   1) 다운받은 원본 데이터(open.zip 압축 푼 폴더)를
#      저장소의 data/ 폴더 구조에 맞게 복사
#   2) 용량 큰 csv들은 상위 500행짜리 _sample.csv 도 별도 생성
#      (Claude와 대화할 때 업로드용으로 사용)
#
# 사용법:
#   ./setup_data.sh <원본데이터_경로> <저장소_경로>
#
# 예시:
#   ./setup_data.sh ~/Downloads/open ~/projects/wind_power
# ==========================================================

set -e  # 에러 나면 즉시 중단

SRC="$1"
REPO="$2"

if [ -z "$SRC" ] || [ -z "$REPO" ]; then
  echo "사용법: ./setup_data.sh <원본데이터_경로> <저장소_경로>"
  echo "예시:   ./setup_data.sh ~/Downloads/open ~/projects/wind_power"
  exit 1
fi

if [ ! -d "$SRC" ]; then
  echo "원본 데이터 경로를 찾을 수 없습니다: $SRC"
  exit 1
fi

if [ ! -d "$REPO" ]; then
  echo "저장소 경로를 찾을 수 없습니다: $REPO"
  exit 1
fi

DEST="$REPO/data"
SAMPLE_DIR="$REPO/data/samples_for_claude"

echo "1) 데이터 폴더 구조 생성..."
mkdir -p "$DEST/train" "$DEST/test" "$SAMPLE_DIR"

echo "2) 원본 데이터 복사..."

# train 파일들
for f in ldaps_train.csv gfs_train.csv train_labels.csv scada_vestas_train.csv scada_unison_train.csv; do
  if [ -f "$SRC/train/$f" ]; then
    cp "$SRC/train/$f" "$DEST/train/$f"
    echo "   복사됨: train/$f"
  else
    echo "   경고: $SRC/train/$f 를 찾지 못했습니다 (건너뜀)"
  fi
done

# test 파일들
for f in ldaps_test.csv gfs_test.csv; do
  if [ -f "$SRC/test/$f" ]; then
    cp "$SRC/test/$f" "$DEST/test/$f"
    echo "   복사됨: test/$f"
  else
    echo "   경고: $SRC/test/$f 를 찾지 못했습니다 (건너뜀)"
  fi
done

# 최상위 파일들
for f in sample_submission.csv info.xlsx data_description.md; do
  if [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DEST/$f"
    echo "   복사됨: $f"
  else
    echo "   경고: $SRC/$f 를 찾지 못했습니다 (건너뜀)"
  fi
done

echo ""
echo "3) 업로드용 샘플 파일 생성 (상위 500행)..."

sample_csv() {
  local relpath="$1"
  local infile="$DEST/$relpath"
  local outname
  outname=$(basename "$relpath" .csv)_sample.csv
  if [ -f "$infile" ]; then
    head -n 500 "$infile" > "$SAMPLE_DIR/$outname"
    echo "   생성됨: data/samples_for_claude/$outname"
  fi
}

sample_csv "train/ldaps_train.csv"
sample_csv "train/gfs_train.csv"
sample_csv "train/train_labels.csv"
sample_csv "train/scada_vestas_train.csv"
sample_csv "train/scada_unison_train.csv"
sample_csv "test/ldaps_test.csv"
sample_csv "test/gfs_test.csv"

# 원본 그대로 복사(가벼운 파일들)
for f in sample_submission.csv info.xlsx data_description.md; do
  if [ -f "$DEST/$f" ]; then
    cp "$DEST/$f" "$SAMPLE_DIR/$f"
    echo "   복사됨(원본): data/samples_for_claude/$f"
  fi
done

echo ""
echo "완료."
echo "  - 전체 데이터: $DEST"
echo "  - 업로드용 샘플: $SAMPLE_DIR (여기 있는 파일들을 Claude 채팅창에 업로드)"
