public class KMP {

    public int getIndex(String str1, String str2) {
        if (str1.length() < str2.length()) {
            return -1;
        }
        char[] ch1 = str1.toCharArray();
        char[] ch2 = str2.toCharArray();
        int[] next = getNextArray(ch2);
        int i = 0, j = 0;
        while (i < ch1.length && j < ch2.length) {
            if (ch1[i] == ch2[j]) {
                i++;
                j++;
            } else if (next[j] == -1) {
                i++;
            } else {
                j = next[j];
            }
        }
        return j == ch2.length ? i - j : -1;
    }

    public static List<Integer> getIndexList(String str1, String str2) {
        List<Integer> list = new ArrayList<>();
        if (str1.length() < str2.length()) {
            list.add(-1);
            return list;
        } else {
            char[] ch1 = str1.toCharArray();
            char[] ch2 = str2.toCharArray();
            int[] next = getNextArray(ch2);
            int i = 0, j = 0;
            while (i < ch1.length) {
                if (ch1[i] == ch2[j]) {
                    i++;
                    j++;
                } else if (next[j] == -1) {
                    i++;
                } else {
                    j = next[j];
                }
                if (j == ch2.length) {
                    list.add(i - j);
                    j = 0;
                }
            }
            if (list.size() == 0) {
                list.add(-1);
            }
            return list;
        }
    }

    public static int[] getNextArray(char[] str) {
        if (str.length < 2) {
            return new int[]{-1};
        }
        int[] next = new int[str.length];
        next[0] = -1;
        next[1] = 0;
        int i = 2;
        int cn = 0;//
        while (i < str.length) {
            //采用递推的思想判断chars[i - 1] chars[cn]是否相等
            if (str[i - 1] == str[cn]) {
                next[i++] = ++cn;
            } else if (cn > 0) {
                cn = next[cn];
            } else {
                next[i++] = 0;
            }
        }
        return next;
    }
}