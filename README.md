# Next.js Error Analysis with Daft.ai 🚀

A comprehensive demo showcasing **Daft.ai's multimodal capabilities** for analyzing Next.js errors using custom UDFs, embedding generation, and RAG queries.

## 🎯 What This Demo Shows

- **📊 Multimodal Analysis**: Process both text logs and image screenshots
- **🔧 Custom UDFs**: Generate embeddings using User-Defined Functions
- **🔍 RAG Queries**: Retrieval-Augmented Generation without vector databases
- **📈 Error Pattern Analysis**: Classify and analyze error types
- **🎨 Embedding Similarity**: Cross-modal similarity search
- **💡 Actionable Insights**: Real recommendations based on error patterns

## 📁 Project Structure

```
daft-nextjsdemo/
├── nextjs_error_analysis.py    # Main analysis script
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── ../nextjs_errors/          # Error data
    ├── logs/                  # Text error logs (10 files)
    │   ├── api_auth_error.txt
    │   ├── build_error.txt
    │   ├── hydration_error.txt
    │   ├── runtime_error.txt
    │   └── ...
    └── screenshots/           # Error screenshots (6 files)
        ├── approutererror.png
        ├── hydrationerror.png
        ├── runtimeerror.png
        └── ...
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Demo

```bash
python3 nextjs_error_analysis.py
```

## 🔍 What the Demo Does

### 1. **Data Loading**

- Loads 13 error entries (7 text logs + 6 image screenshots)
- Processes real Next.js error scenarios

### 2. **UDF Embedding Generation**

- **Text UDFs**: Generate 512-dimensional embeddings from error logs
- **Image UDFs**: Generate embeddings from error screenshots
- **Custom Logic**: Different embedding patterns for different error types

### 3. **Error Classification**

- Authentication Errors
- Hydration Errors
- Runtime Errors
- Build Errors
- Module Errors
- Styling Errors

### 4. **RAG Queries**

- Find authentication-related errors
- Locate visual error screenshots
- Search for runtime issues
- Perform multimodal similarity search

### 5. **Actionable Insights**

- Error type distribution analysis
- Specific recommendations for each error category
- Performance metrics and benefits

## 📊 Sample Output

```
🎯 NEXT.JS ERROR ANALYSIS WITH DAFT.AI
==================================================

🚀 Setting up Daft.ai...
✅ Ready for multimodal analysis!

📁 Loading Next.js error data...
✅ Loaded 13 error entries

📊 Creating DataFrame with embeddings...
✅ Generated text embeddings using UDF
✅ Generated image embeddings using UDF

🔍 EMBEDDING VERIFICATION:
✅ Text embeddings: 13 entries
✅ Image embeddings: 6 entries
   Text embedding dimension: 512
   Image embedding dimension: 512

📈 ERROR TYPE DISTRIBUTION:
- Authentication Error: 3 occurrences
- Hydration Error: 2 occurrences
- Runtime Error: 2 occurrences
- Build Error: 2 occurrences

🎯 RECOMMENDATIONS:
🔐 Authentication Issues (3 errors):
   → Review API key configuration
   → Check authentication middleware setup
```

## 🚀 Key Features Demonstrated

### ✅ **Custom UDFs**

```python
@daft.udf(return_dtype=daft.DataType.list(daft.DataType.float64()))
def generate_text_embedding(text_content: daft.Series) -> daft.Series:
    # Custom embedding logic based on error content
    return [create_text_embedding(text) for text in text_content]
```

### ✅ **Multimodal Processing**

- Text logs → Text embeddings
- Image screenshots → Image embeddings
- Cross-modal similarity search

### ✅ **RAG Without Vector DB**

- All embeddings stored in DataFrame
- No external vector database needed
- Fast similarity queries using Daft operations

### ✅ **Distributed Processing**

- Uses Daft's distributed execution
- Scales with data size
- Efficient UDF execution

## 🎯 Demo Highlights

1. **🔧 UDF Power**: Custom embedding generation functions
2. **📊 Multimodal**: Text + Image analysis together
3. **🔍 RAG Queries**: Intelligent error retrieval
4. **💡 Insights**: Actionable recommendations
5. **🚀 Performance**: Process 13 errors in seconds

## 📋 Error Types Analyzed

| Error Type     | Count | Files                                            |
| -------------- | ----- | ------------------------------------------------ |
| Authentication | 3     | `api_auth_error.txt`, `openaierror.png`          |
| Hydration      | 2     | `hydration_error.txt`, `hydrationerror.png`      |
| Runtime        | 2     | `runtime_error.txt`, `runtimeerror.png`          |
| Build          | 2     | `build_error.txt`                                |
| Module         | 1     | `module_not_found.txt`, `module.png`             |
| Styling        | 1     | `tailwind_config_error.txt`, `tailwinderror.png` |

## 🔧 Technical Details

- **Daft.ai Version**: 0.5.10
- **Embedding Dimensions**: 512
- **Processing Time**: < 5 seconds
- **Memory Efficient**: No external vector storage
- **Scalable**: Distributed UDF execution

## 🎉 Benefits Showcased

- ✅ **No Vector Database**: Everything in DataFrame
- ✅ **Custom UDFs**: Write your own functions
- ✅ **Multimodal**: Text + Images together
- ✅ **Fast Processing**: Seconds, not minutes
- ✅ **Actionable**: Real recommendations

## 🚀 Next Steps

This demo can be extended to:

- Real embedding models (OpenAI, Hugging Face)
- More complex error patterns
- Live error monitoring
- Automated recommendations
- Integration with CI/CD pipelines

## 📝 License

MIT License - Feel free to use and modify for your projects!

---

**Built with ❤️ using [Daft.ai](https://www.getdaft.io/) - The fastest DataFrame library for Python**
