import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class DataPrepare {
    String csvFile = "train.csv";
    String outCsvFile = "train_out.csv";
    BufferedReader br = null;
    String line = "";
    String cvsSplitBy = ",";
    PrintWriter pw = null;

    private void ReadFile() {
        try {

            br = new BufferedReader(new FileReader(csvFile));
            pw = new PrintWriter(new File(outCsvFile));
            line = br.readLine();// skip header
            pw.write(line + System.getProperty("line.separator"));
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(cvsSplitBy);
                String ageuponOutcome = columns[7];
                if(ageuponOutcome == null || ageuponOutcome.isEmpty()){
                    // give default 0 years to empty rows
                    ageuponOutcome = "0 years";
                }
                String[] ageuponOutcomes = ageuponOutcome.split(" ");
                int age = Integer.parseInt( ageuponOutcomes[0]);
                double multiplier = 1;
                if(age == 0){
                    // give default half value
                    multiplier = 0.5;
                    age = 1;
                }
                String unit = ageuponOutcomes[1];
                if(unit.startsWith("day")){

                }else if(unit.startsWith("week")){
                    age = (int)(age * 7 * multiplier);
                }else if(unit.startsWith("month")){
                    age = (int)(age * 30 * multiplier);
                }else if(unit.startsWith("year")){
                    age = (int)(age * 365 * multiplier);
                }
                pw.write(String.join(cvsSplitBy, columns) + "," + age + System.getProperty("line.separator"));
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(pw != null){
                try{
                    pw.close();
                }catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
    }

    public void Run() {
        ReadFile();
    }

    public static void main(String[] args) {
        DataPrepare dp = new DataPrepare();
        dp.Run();
    }
}