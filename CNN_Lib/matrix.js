export default class Matrix {
  rows;
  columns;
  data;

  constructor(numRows, numColumns) {
    this.rows = numRows;
    this.columns = numColumns;

    this.initMatrixData();
  }

  static fromArrayToVector(arr) {
    let resultMatrix = new Matrix(arr.length, 1);

    for (let i = 0; i < resultMatrix.rows; i++) {
      for (let j = 0; j < resultMatrix.columns; j++) {
        resultMatrix.data[i][j] = arr[i];
      }
    }

    return resultMatrix;
  }

  static fromArrayToMatrix(arr, rows, cols) {
    let resultMatrix = new Matrix(rows, cols);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        resultMatrix.data[i][j] = arr[i * rows + j];
      }
    }

    return resultMatrix;
  }

  static toArray(m) {
    if (m instanceof Matrix) {
      return m.data.flatMap((val) => val);
    } else {
      return m.flatMap((val) => val);
    }
  }

  static multiply(m1, m2) {
    if (m1.columns !== m2.rows) return;

    let resultMatrix = new Matrix(m1.rows, m2.columns);

    for (let i = 0; i < resultMatrix.rows; i++) {
      for (let j = 0; j < resultMatrix.columns; j++) {
        let sum = 0;
        for (let k = 0; k < m1.columns; k++) {
          sum += m1.data[i][k] * m2.data[k][j];
        }
        resultMatrix.data[i][j] = sum;
      }
    }

    return resultMatrix;
  }

  static subtract(m1, m2) {
    let resultMatrix = new Matrix(m1.rows, m1.columns);

    for (let i = 0; i < resultMatrix.rows; i++) {
      for (let j = 0; j < resultMatrix.columns; j++) {
        resultMatrix.data[i][j] = m1.data[i][j] - m2.data[i][j];
      }
    }

    return resultMatrix;
  }

  static map(m, func) {
    let resultMatrix = new Matrix(m.rows, m.columns);

    for (let i = 0; i < resultMatrix.rows; i++) {
      for (let j = 0; j < resultMatrix.columns; j++) {
        const val = m.data[i][j];

        resultMatrix.data[i][j] = func(val);
      }
    }

    return resultMatrix;
  }

  static transpose(m) {
    let resultMatrix = new Matrix(m.columns, m.rows);

    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.columns; j++) {
        resultMatrix.data[j][i] = m.data[i][j];
      }
    }

    return resultMatrix;
  }

  static reverseCols(m) {
    let resultMatrix = new Matrix(m.rows, m.columns);

    for (let i = 0; i < m.rows; i++) {
      for (let j = 0, k = m.columns - 1; j < m.columns; j++, k--) {
        resultMatrix.data[i][j] = m.data[i][k];
      }
    }

    return resultMatrix;
  }

  static convolve(m1, m2) {
    // if (m1.rows !== m2.rows && m1.columns !== m2.columns) return;

    let sum = 0;

    for (let i = 0; i < m1.rows; i++) {
      for (let j = 0; j < m1.columns; j++) {
        sum += m1.data[i][j] * m2.data[i][j];
      }
    }

    return sum;
  }

  static sumMatrixValues(m) {
    let sum = 0;

    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.columns; j++) {
        sum += m.data[i][j];
      }
    }

    return sum;
  }

  add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  subtract(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] -= n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] -= n;
        }
      }
    }
  }

  multiply(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] *= n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] *= n;
        }
      }
    }
  }

  divide(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] /= n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] /= n;
        }
      }
    }
  }

  map(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        const val = this.data[i][j];

        this.data[i][j] = func(val);
      }
    }
  }

  rotate180() {
    let matrix180 = Matrix.transpose(this);
    matrix180 = Matrix.reverseCols(matrix180);
    matrix180 = Matrix.transpose(matrix180);
    matrix180 = Matrix.reverseCols(matrix180);

    return matrix180;
  }

  static pad(m, pRows, pCols) {
    let resultMatrix = m;

    for (let i = 0; i < resultMatrix.rows; i++) {
      for (let j = 0; j < pCols / 2; j++) {
        resultMatrix.data[i].unshift(0);
        resultMatrix.data[i].push(0);
      }
    }

    let padRow = new Array(resultMatrix.columns + pCols).fill(0);
    for (let i = 0; i < pRows; i += pRows / 2) {
      resultMatrix.data.unshift(padRow);
      resultMatrix.data.push(padRow);
    }

    resultMatrix.setRows(resultMatrix.rows + pRows);
    resultMatrix.setColumns(resultMatrix.columns + pCols);

    return resultMatrix;
  }

  initMatrixData() {
    let matrix = [];

    for (let i = 0; i < this.rows; i++) {
      matrix[i] = [];
      for (let j = 0; j < this.columns; j++) {
        matrix[i][j] = this.generateRandomVal(-0.5, 0.5);
      }
    }

    this.data = matrix;
  }

  generateRandomVal(min, max) {
    return Math.random() * (max - min) + min;
  }

  setRows(rows) {
    this.rows = rows;
  }

  setColumns(columns) {
    this.columns = columns;
  }

  static findMaxElseZero = function (chunkArr) {
    let xMax = 0;
    let yMax = 0;

    let modArr = [];
    let maxArr = [];
    for (let h = 0; h < chunkArr.length; h++) {
      let chunk = chunkArr[h];
      let max = chunk.data[0][0];
      for (let i = 0; i < chunk.data.length; i++) {
        for (let j = 0; j < chunk.data[0].length; j++) {
          if (chunk.data[i][j] > max) {
            max = chunk.data[i][j];
            chunk.data[xMax][yMax] = 0;
            xMax = i;
            yMax = j;
          } else {
            chunk.data[i][j] = 0;
          }
        }
      }
      maxArr.push(max);
      modArr.push(chunk);
    }

    return { maxArr, modArr };
  };

  static deconstructMatrix(arr, stride, rows, cols) {
    const mChunkLength =
      ((arr.length - rows) / stride + 1) *
      ((arr[0].length - cols) / stride + 1);

    const mChunkArr = new Array(mChunkLength);

    let mChunkIndex = 0;

    outer: for (let i = 0; i < arr.length; i += stride) {
      const mStride = arr.slice(i, rows + i);

      for (let j = 0; j < mStride[0].length - (rows - 1); j += stride) {
        let matrix = new Matrix(rows, cols);

        for (let k = 0; k < cols; k++) {
          let mSlice = mStride[k].slice(j, cols + j);

          for (let l = 0; l < mSlice.length; l++) {
            matrix.data[k][l] = mSlice[l];
          }
        }

        mChunkArr[mChunkIndex] = matrix;
        mChunkIndex++;

        if (mChunkIndex === mChunkLength) break outer;
      }
    }

    return mChunkArr;
  }

  static reconstructMatrix(chunkArr, stride, rows, cols) {
    const reconst = new Matrix(rows, cols);

    for (let i = 0; i < rows; i += stride) {
      const chunk = chunkArr.slice(i, rows / stride + i);

      for (let j = 0; j < chunk.length; j++) {
        const chunkSlice = chunk[j].data;

        for (let k = 0; k < stride; k++) {
          const slice = chunkSlice[k].slice(0, stride);

          for (let l = 0; l < slice.length; l++) {
            reconst.data[k + i][j * stride + l] = slice[l];
          }
        }
      }
    }

    return reconst;
  }
}
