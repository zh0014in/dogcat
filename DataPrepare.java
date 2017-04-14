import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.time.format.*;
import java.text.*;
import java.util.*;

public class DataPrepare {
    String csvFile = "train.csv";
    String outCsvFile = "train_out.csv";
    BufferedReader br = null;
    String line = "";
    String csvSplitBy = ",";
    PrintWriter pw = null;
    int indexShift = 0;
    public DataPrepare(){

    }

    public DataPrepare(String inputFile, String outputFile, int indexShift){
        this.csvFile = inputFile;
        this.outCsvFile = outputFile;
        this.indexShift = indexShift;
    }

    private void ReadFile() {
        try {

            br = new BufferedReader(new FileReader(csvFile));
            pw = new PrintWriter(new File(outCsvFile));
            line = br.readLine();// skip header
            pw.write(line + ",age,hours,week,month,sex1,sex2,isMix,simpleColor,hasName,type"
                    + System.getProperty("line.separator"));
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(csvSplitBy);
                int age = ParseAgeuponOutcome(columns);
                int hours = ParseDateTimeHourofDay(columns);
                int week = ParseDatetimeWeek(columns);
                int month = ParseDatetimeMonth(columns);
                int sex1 = ParseSexuponOutcome(columns);
                int sex2 = ParseSexuponOutcome2(columns);
                int mix = ParseIsMix(columns);
                int color = ParseColor(columns);
                int hasName = ParseName(columns);
                int type = ParseType(columns);
                pw.write(String.join(csvSplitBy, columns) + "," + age + "," + hours + "," + week + "," + month + ","
                        + sex1 + "," + sex2 + "," + mix + "," + color + "," + hasName + "," + type
                        + System.getProperty("line.separator"));
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
            if (pw != null) {
                try {
                    pw.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private int ParseAgeuponOutcome(String[] columns) {
        String ageuponOutcome = columns[7+indexShift];
        if (ageuponOutcome == null || ageuponOutcome.isEmpty()) {
            // give default 0 years to empty rows
            ageuponOutcome = "0 years";
        }
        String[] ageuponOutcomes = ageuponOutcome.split(" ");
        int age = Integer.parseInt(ageuponOutcomes[0]);
        double multiplier = 1;
        if (age == 0) {
            // give default half value
            multiplier = 0.5;
            age = 1;
        }
        String unit = ageuponOutcomes[1];
        if (unit.startsWith("day")) {

        } else if (unit.startsWith("week")) {
            age = (int) (age * multiplier);
        } else if (unit.startsWith("month")) {
            age = (int) (age * 4 * multiplier);
        } else if (unit.startsWith("year")) {
            age = (int) (age * 52 * multiplier);
        }
        return age;
    }

    private int ParseDateTimeHourofDay(String[] columns) {
        String dateTime = columns[2];
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        try {
            Date date = simpleDateFormat.parse(dateTime);

            Calendar cal = Calendar.getInstance();
            cal.setTime(date);
            int hours = cal.get(Calendar.HOUR_OF_DAY);
            return hours;
        } catch (ParseException ex) {
            System.out.println("Exception " + ex);
        }
        return 0;
    }

    private int ParseDatetimeWeek(String[] columns) {
        String dateTime = columns[2];
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        try {
            Date date = simpleDateFormat.parse(dateTime);

            Calendar cal = Calendar.getInstance();
            cal.setTime(date);
            int week = cal.get(Calendar.WEEK_OF_MONTH);
            return week;
        } catch (ParseException ex) {
            System.out.println("Exception " + ex);
        }
        return 0;
    }

    private int ParseDatetimeMonth(String[] columns) {
        String dateTime = columns[2];
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        try {
            Date date = simpleDateFormat.parse(dateTime);

            Calendar cal = Calendar.getInstance();
            cal.setTime(date);
            int week = cal.get(Calendar.MONTH);
            return week;
        } catch (ParseException ex) {
            System.out.println("Exception " + ex);
        }
        return 0;
    }

    private int ParseSexuponOutcome(String[] columns) {
        String sex = columns[6+indexShift];
        if (sex.toLowerCase().equals("unknown")) {
            return -1;
        }
        String[] parts = sex.split(" ");
        if (parts[0].toLowerCase().equals("spayed") || parts[0].toLowerCase().equals("neutered")) {
            return 1;
        }
        return 0;
    }

    private int ParseSexuponOutcome2(String[] columns) {
        String sex = columns[6+indexShift];
        if (sex.toLowerCase().equals("unknown")
        || sex == null || sex.isEmpty()) {
            return -1;
        }
        String[] parts = sex.split(" ");
        if (parts[1].toLowerCase().equals("male")) {
            return 1;
        }
        return 0;
    }

    private int ParseIsMix(String[] columns) {
        String breed = columns[8+indexShift];
        if (breed.toLowerCase().contains("mix")) {
            return 1;
        }
        return 0;
    }

    private int ParseColor(String[] columns) {
        String color = columns[9+indexShift];
        if (color.contains("/") || color.contains(" ")) {
            return 1;
        }
        return 0;
    }

    private int ParseName(String[] columns) {
        String name = columns[1];
        if (name != null && !name.isEmpty()) {
            return 1;
        }
        return 0;
    }

    private int ParseType(String[] columns){
        String type = columns[5+indexShift];
        if(type.equals("Dog")){
            return 1;
        }
        return 0;
    }

    public void Run() {
        ReadFile();
    }

    public static void main(String[] args) {
        DataPrepare dp = new DataPrepare();
        dp.Run();
        dp = new DataPrepare("test.csv", "test_out.csv", -2);
        dp.Run();
    }
}