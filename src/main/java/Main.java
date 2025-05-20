import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import com.google.gson.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.*;

public class Main {
    static String[] corpusFiles = {
      "corpus_20mb_part_0.jsonl", "corpus_20mb_part_1.jsonl", "corpus_20mb_part_2.jsonl",
      "corpus_20mb_part_3.jsonl", "corpus_20mb_part_4.jsonl", "corpus_20mb_part_5.jsonl",
      "corpus_20mb_part_6.jsonl", "corpus_20mb_part_7.jsonl", "corpus_20mb_part_8.jsonl",
      "corpus_20mb_part_9.jsonl", "corpus_20mb_part_10.jsonl", "corpus_20mb_part_11.jsonl",
      "corpus_20mb_part_12.jsonl"
    };

    static class QueryObject {
        String _id;
        String text;
    }

    static class DocResult {
        String docId;
        int rank;
        double score;
        DocResult(String docId, int rank, double score) {
            this.docId = docId;
            this.rank = rank;
            this.score = score;
        }
    }

    public static void main(String[] args) throws Exception {
        String queryPath = "queries.jsonl";
        String judgmentPath = "test.tsv";

        Directory dir = FSDirectory.open(Paths.get("index"));
        Analyzer analyzer = new EnglishAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(dir, config);
        Gson gson = new Gson();

        // 1. INDEXLEME
        for (String fileName : corpusFiles) {
            Path path = Paths.get(fileName);
            if (!Files.exists(path)) {
                System.out.println("Dosya bulunamadı: " + fileName);
                continue;
            }
            List<String> lines = Files.readAllLines(path);
            for (String line : lines) {
                JsonObject obj = gson.fromJson(line, JsonObject.class);
                Document doc = new Document();
                doc.add(new StringField("id", obj.get("_id").getAsString(), Field.Store.YES));
                doc.add(new TextField("text", obj.get("text").getAsString().toLowerCase(), Field.Store.YES));
                writer.addDocument(doc);
            }
        }
        writer.close();
        System.out.println("Indexleme tamamlandı.");

        // 2. QUERIES'i oku
        List<QueryObject> queries = Files.lines(Paths.get(queryPath))
            .map(line -> gson.fromJson(line, QueryObject.class))
            .collect(Collectors.toList());

        // 3. BM25 ve LMDirichlet ile arama ve run dosyaları oluşturma
        searchAndWriteRunFile(dir, analyzer, queries, new BM25Similarity(), "bm25.run", "bm25");
        searchAndWriteRunFile(dir, analyzer, queries, new LMDirichletSimilarity(), "lmd.run", "lmd");

        // 4. test.tsv'den relevance'ları oku
        Map<String, Map<String, Integer>> qrels = loadJudgments(judgmentPath);

        // 5. nDCG hesapla
        calculateNDCG("bm25.run", qrels, "BM25");
        calculateNDCG("lmd.run", qrels, "LMDirichlet");
    }

    static void searchAndWriteRunFile(Directory dir, Analyzer analyzer, List<QueryObject> queries,
                                      Similarity similarity, String runFileName, String runTag) throws Exception {
        IndexReader reader = DirectoryReader.open(dir);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(similarity);
        QueryParser parser = new QueryParser("text", analyzer);
        BufferedWriter writer = new BufferedWriter(new FileWriter(runFileName));

        for (QueryObject q : queries) {
            Query luceneQuery = parser.parse(QueryParser.escape(q.text.toLowerCase()));
            TopDocs topDocs = searcher.search(luceneQuery, 100);
            ScoreDoc[] hits = topDocs.scoreDocs;
            for (int rank = 0; rank < hits.length; rank++) {
                Document doc = searcher.doc(hits[rank].doc);
                writer.write(String.format("%s Q0 %s %d %.4f %s\n",
                        q._id, doc.get("id"), rank + 1, hits[rank].score, runTag));
            }
        }
        writer.close();
        System.out.println(runFileName + " oluşturuldu.");
    }

    static Map<String, Map<String, Integer>> loadJudgments(String path) throws IOException {
        Map<String, Map<String, Integer>> qrels = new HashMap<>();
        List<String> lines = Files.readAllLines(Paths.get(path));

        for (String line : lines) {
            if (line.trim().toLowerCase().startsWith("topic") || line.trim().isEmpty()) continue;

            String[] parts = line.split("\t");
            if (parts.length < 3) continue;

            try {
                String topic = parts[0];
                String docId = parts[1];
                int rel = Integer.parseInt(parts[2]);

                qrels.putIfAbsent(topic, new HashMap<>());
                qrels.get(topic).put(docId, rel);
            } catch (NumberFormatException e) {
                System.err.println("Hatalı satır atlandı: " + line);
            }
        }
        return qrels;
    }

    static void calculateNDCG(String runFile, Map<String, Map<String, Integer>> qrels, String runName) throws IOException {
        Map<String, List<DocResult>> results = new HashMap<>();
        for (String line : Files.readAllLines(Paths.get(runFile))) {
            String[] parts = line.split(" ");
            String topic = parts[0];
            String docId = parts[2];
            int rank = Integer.parseInt(parts[3]);
            double score = Double.parseDouble(parts[4]);

            results.putIfAbsent(topic, new ArrayList<>());
            results.get(topic).add(new DocResult(docId, rank, score));
        }

        double total10 = 0.0, total100 = 0.0;
        int count = 0;

        for (String topic : results.keySet()) {
            List<DocResult> docs = results.get(topic);
            docs.sort(Comparator.comparingInt(d -> d.rank));

            List<Integer> rels = docs.stream()
                    .map(d -> qrels.getOrDefault(topic, new HashMap<>()).getOrDefault(d.docId, 0))
                    .collect(Collectors.toList());

            total10 += computeNDCG(rels, 10);
            total100 += computeNDCG(rels, 100);
            count++;
        }

        System.out.printf("[%s] Avg nDCG@10: %.4f | Avg nDCG@100: %.4f\n", runName, total10 / count, total100 / count);
    }

    static double computeNDCG(List<Integer> rels, int k) {
        double dcg = 0.0, idcg = 0.0;
        for (int i = 0; i < k && i < rels.size(); i++) {
            int rel = rels.get(i);
            dcg += (Math.pow(2, rel) - 1) / (Math.log(i + 2) / Math.log(2));
        }

        List<Integer> sorted = new ArrayList<>(rels);
        sorted.sort(Comparator.reverseOrder());

        for (int i = 0; i < k && i < sorted.size(); i++) {
            int rel = sorted.get(i);
            idcg += (Math.pow(2, rel) - 1) / (Math.log(i + 2) / Math.log(2));
        }

        return idcg == 0 ? 0 : dcg / idcg;
    }
}
