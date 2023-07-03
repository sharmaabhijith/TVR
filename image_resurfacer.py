# Importing general libraries
import numpy as np

# Importing library to calculate the total variation loss (NOTE: ensure installation first)
from torchmetrics import TotalVariation

class Total_Variation_Resurfacer:
    """Total Variation Based Image Resurfacing (TVR) to cleanse image of arbitrary number of adversarial patches"""
    
    def __init__(self, image, block_shape):
        self.image = image  # The image to be defended (@: [3*224*224] tensor)
        self.image_length = image.shape[-2]  # Length of the image (@: 224)
        self.image_width = image.shape[-1]  # Width of the image (@: 224)
        # The length and width of each block in Image Block-set
        block_length = block_shape[0]
        block_width = block_shape[1]
        if (self.image_length%block_length != 0):
            raise ValueError(f"Block-length: {block_length} must be a factor of Image-width: {self.image_length} for pure Block-set")
        else:
            self.block_length = block_length
        if (self.image_width%block_width != 0):
            raise ValueError(f"Block-width: {block_width} must be a factor of Image-width: {self.image_width} for pure Block-set")
        else:
            self.block_width = block_width
        # Number of blocks in the Image Block-set (instance member)
        self.image_area = self.image_length*self.image_width
        self.block_area = self.block_length*self.block_width
        self.num_blocks = int(image_area/block_area)


    # STAGE: 1
    def Image_to_Block(self):
        # Number of blocks per row and column
        num_blocks_per_row = int(self.image_width/self.block_width)
        num_blocks_per_col = int(self.image_length/self.block_length)
        """Convert the image into a channel-wise Block-set"""
        # NOTE: Each block acts as Neighbourhood in TV-loss calculation
        # Channel-wise dummy Block-set generation
        self.red_blockset = torch.zeros((num_blocks, self.block_length, self.block_width))
        self.green_blockset = torch.zeros((num_blocks, self.block_length, self.block_width))
        self.blue_blockset = torch.zeros((num_blocks, self.block_length, self.block_width))

        # Channel-wise Image Block-set generation
        for b in range(self.num_blocks):
            for i in range(self.block_length):
                for j in range(self.block_width):
                    # NOTE: Multiplied by 255 to convert image from [0, 1] to [0, 255] for better TV-loss calculation
                    self.red_blockset[b][i][j] = int(self.image[0][0][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j]*255)
                    self.green_blockset[b][i][j] = int(self.image[0][1][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j]*255)
                    self.blue_blockset[b][i][j] = int(self.image[0][2][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j]*255)


    # STAGE: 2
    def calculate_TV_Score(self):
        """Calculate total variation loss over each block in the Image Block-set"""
        tv = TotalVariation() # Off-the-shelf total variation loss from  torchmetrics 
        self.red_tvlist = []
        self.green_tvlist = []
        self.blue_tvlist = []
        # Check if channel-wise Block-set is available
        if (self.red_blockset==None):
            raise AttributeError("Generate red channel Block set first from the image")
        if (self.green_blockset==None):
            raise AttributeError("Generate green channel Block set first from the image")
        if (self.blue_blockset==None):
            raise AttributeError("Generate blue channel Block set first from the image")
        else:
            for blk in range(self.num_blocks):
                # Calculate TV-loss/score over each block and store in $x$_tvlist for each channel
                # RED
                tv_score = tv(self.red_blockset[blk].unsqueeze(0).unsqueeze(0))
                self.red_tvlist.append(tv_score)
                # GREEN
                tv_score = tv(self.green_blockset[blk].unsqueeze(0).unsqueeze(0))
                self.green_tvlist.append(tv_score)
                # BLUE
                tv_score = tv(self.blue_blockset[blk].unsqueeze(0).unsqueeze(0))
                self.blue_tvlist.append(tv_score)
                
            self.red_tvlist, self.green_tvlist, self.blue_tvlist = np.array(self.red_tvlist), np.array(self.green_tvlist), np.array(self.blue_tvlist)


    # STAGE: 3
    def outlier_Detection(self):
        """Detect channel-wise outlier blocks based on the TV score in the Image Block-set"""
        # Percentile for Inter-quartile range
        low_percentile = 25
        high_percentile = 75
        # Flagging those blocks whose TV-loss is greater than the outlier of TV-loss over the entire Block-set
        # $xx_outlierflag$ returns the list of those  indexes where block's TV score is higher than the outlier
        # RED
        Q1 = np.percentile(self.red_tvlist, low_percentile)
        Q3 = np.percentile(self.red_tvlist, high_percentile)
        IQR = Q3 - Q1
        upper = Q3 + 1.5*IQR
        red_outlierflag = np.where(self.red_tvlist > upper)[0]
        # GREEN
        Q1 = np.percentile(self.green_tvlist, low_percentile)
        Q3 = np.percentile(self.green_tvlist, high_percentile)
        IQR = Q3 - Q1
        upper = Q3 + 1.5*IQR  
        green_outlierflag = np.where(self.green_tvlist > upper)[0]
        # BLUE
        Q1 = np.percentile(self.blue_tvlist, low_percentile)
        Q3 = np.percentile(self.blue_tvlist, high_percentile)
        IQR = Q3 - Q1
        upper = Q3 + 1.5*IQR
        blue_outlierflag = np.where(self.blue_tvlist > upper)[0]


    # STAGE: 4
    def obsfucated_Image(self):
        """create obsfucated(called as cropped image in paper) from Image Block-set based on outliers.
        This function also creates a mask for producing regenerated image"""
        # Initialize dummy mask and cropped image
        self.mask =  torch.zeros((1, 3, self.image_length, self.image_width))
        self.cropped_image = torch.zeros((1, 3, self.image_length, self.image_width))
        # Number of blocks per row and column
        num_blocks_per_row = int(self.image_width/self.block_width)
        num_blocks_per_col = int(self.image_length/self.block_length)
        # Image generation from channel-wise Image Block-set
        for b in range(self.num_blocks):
            flag=0
            for i in range(self.block_length):
                for j in range(self.block_width):
                    # NOTE: Divide by 255 to convert image from [0, 255] to [0, 1] back to original form
                    # RED
                    if b in red_outlierflag:
                        flag=1
                    else:
                        self.cropped_image[0][0][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] \
                            = self.red_blockset[b][i][j]/255
                    # GREEN
                    if b in green_outlier_flag:
                        flag=1
                    else:
                        self.cropped_image[0][1][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] \
                            = self.green_blockset[b][i][j]/255
                    # BLUE
                    if b not in blue_outlierflag:
                        flag=1
                    else:
                        self.cropped_image[0][2][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] \
                            = self.blue_blockset[b][i][j]/255

                    # Create Mask based on the cropped image
                    if flag==1:
                        self.mask[0][0][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] = 1
                        self.mask[0][1][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] = 1
                        self.mask[0][2][int(b/num_blocks_per_col)*self.block_length+i][int(b%num_blocks_per_row)*self.block_width+j] = 1

    # STAGE: 5
    def reconstructed_Image(self, generatorNet):
        """create new reconstructed image from cropped image"""
        # Generator is pre-trained for image inpainting
        self.generated_image = generatorNet(self.cropped_image)
        # Hadamard Product
        self.reconstructed_image = torch.mul((1-self.mask), self.cropped_image) + torch.mul(self.mask, self.generated_image)

        return self.reconstructed_image













