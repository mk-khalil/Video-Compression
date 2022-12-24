import heapq

class HuffCode:
    def __init__(self, file_path = None, data = None):
        # if the input is a variable, pass it to the message argument, 
        # otherwise pass input path to file path 
        self.fpath = file_path
        self.data = data
        #heap to be used for merging nodes
        self.heap = []
        #character to code mapping dictionary
        self.char2code = {}
        #code to character reverse mapping dictionary
        self.code2char = {}

    class Node:
        #nodes to be used in constructing huffman tree
        def __init__(self, char, freq):
            #symbol represented by node
            self.char = char
            #its probability of appearance
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            #comparator definitions for comparing nodes
            return self.freq < other.freq

        def __eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, Node)):
                return False
            return self.freq == other.freq

    def generate_frequencies(self, text):
        #generating dictionary containing frequency(count) of each character
        freq = {}
        for char in text:
            if not char in freq:
                freq[char] = 0
            freq[char] = 1
        return freq

    def generate_heap(self, freq):
        #generating heap used to create the huffman tree
        for key in freq:
            heapq.heappush(self.heap, self.Node(key, freq[key]))

    def node_merger(self):
        #merging probabilities in the huffman tree
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            mergenode = self.Node(None, node1.freq + node2.freq)
            mergenode.left, mergenode.right = node1, node2
            heapq.heappush(self.heap, mergenode)

    def generate_code_rec(self, root, current_code):
        #recursive function that traverses the huffman tree and assigns codes to each node
        if (root == None):
            return

        if (root.char != None):
            self.char2code[root.char] = current_code
            self.code2char[current_code] = root.char
            return

        self.generate_code_rec(root.left, current_code + "0")
        self.generate_code_rec(root.right, current_code + "1")

    def generate_code(self):
        self.generate_code_rec(heapq.heappop(self.heap), "")
    #using the assigned char2code dictionary to encode the text
    def encoder(self, text):
        encoded = ""
        for char in text:
            encoded += self.char2code[char]
        return encoded

    def compress(self):
      #compression function
      #if message is none read a file
      if self.data is None:
        with open(self.fpath, 'r+') as file:
            text = file.read()
            text = text.rstrip()
      else:
            text = self.data
      #generating frequencies dict
      counts = self.generate_frequencies(text)
      #generating node heap
      self.generate_heap(counts)
      #generating char2code dict
      self.node_merger()
      self.generate_code()

      #using char2code to encode text
      encoded = self.encoder(text)

      print("Compressed")
      return encoded,self.code2char

    def decoder(self, encoded, hufftree):
        #using the assigned code2char dictionary to decode the text
        code = ""
        decoded = ""
        for i, bit in enumerate(encoded):
            code += bit
            if code in hufftree:
                char = hufftree[code]
                decoded += char
                code = ""

        return decoded

    def decompress(self,hufftree, input_path = None, data_compressed = None ):
      #decompression function
      if data_compressed is None:
         #if message is none read a file
        with open(input_path, 'r') as file:
            encoded = file.read()
      else:
        encoded = data_compressed
        #decoding message using code2char
        decoded = self.decoder(encoded, hufftree)
        print("Decompressed")
        return decoded