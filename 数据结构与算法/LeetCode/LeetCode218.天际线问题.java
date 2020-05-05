import java.util.*;

/**
 * @author Ning
 * @since 2020.05.01 15:46
 */
public class Skyline {
    public static class Node {
        public boolean isUp;
        public int position;
        public int height;

        public Node(boolean isUp, int position, int height) {
            this.isUp = isUp;
            this.position = position;
            this.height = height;
        }
    }

    public class NodeComparator implements Comparator<Node> {
        @Override
        public int compare(Node o1, Node o2) {
            if (o1.position != o2.position) {
                return o1.position - o2.position;
            }
            if (o1.isUp != o2.isUp) {
                return o1.isUp ? -1 : 1;
            }
            return 0;
        }
    }

    public List<List<Integer>> getSkyline(int[][] buildings) {
        Node[] nodes = new Node[buildings.length * 2];
        for (int i = 0; i < buildings.length; i++) {
            nodes[i * 2] = new Node(true, buildings[i][0], buildings[i][2]);
            nodes[i * 2 + 1] = new Node(false, buildings[i][1], buildings[i][2]);
        }
        Arrays.sort(nodes, new NodeComparator());
        TreeMap<Integer, Integer> htMap = new TreeMap<>();
        TreeMap<Integer, Integer> posiMap = new TreeMap<>();
        for (int i = 0; i < nodes.length; i++) {
            if (nodes[i].isUp) {
                if (!htMap.containsKey(nodes[i].height)) {
                    htMap.put(nodes[i].height, 1);
                } else {
                    htMap.put(nodes[i].height, htMap.get(nodes[i].height) + 1);
                }
            } else {
                if (htMap.containsKey(nodes[i].height)) {
                    if (htMap.get(nodes[i].height) == 1) {
                        htMap.remove(nodes[i].height);
                    } else {
                        htMap.put(nodes[i].height, htMap.get(nodes[i].height) - 1);
                    }
                }
            }
            if (htMap.isEmpty()) {
                posiMap.put(nodes[i].position, 0);
            } else {
                posiMap.put(nodes[i].position, htMap.lastKey());
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        int start = 0;
        int height = 0;
        for (Map.Entry<Integer, Integer> entry : posiMap.entrySet()) {
            int curPosition = entry.getKey();
            int curMaxHeight = entry.getValue();
            if (height != curMaxHeight) {
                start = curPosition;
                height = curMaxHeight;
                List<Integer> list = new ArrayList<>();
                list.add(start);
                list.add(height);
                res.add(list);
            }

        }
        return res;
    }
}